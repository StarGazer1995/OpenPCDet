from numpy.lib.polynomial import polymul
import torch.nn as nn
import torch
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PVNet(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        nsample = self.model_cfg.NSAMPLE
        assert 4 == len(nsample)
        conv_feat_size = 64 + 64*nsample[0] + 32*nsample[0]*nsample[1] + \
            16*nsample[0]*nsample[1]*nsample[2] + 3*nsample[0]*nsample[1]*nsample[2]*nsample[3] + 3

        self.mlps_ = nn.Sequential(
            nn.Conv1d(int(conv_feat_size), int(conv_feat_size), kernel_size=1, bias=False),
            # nn.Linear(int(conv_feat_size), 2 * int(conv_feat_size)),
            nn.BatchNorm1d(int(conv_feat_size)),
            nn.ReLU,
            # nn.Linear(2 * int(conv_feat_size), int(conv_feat_size)),
            nn.Conv1d(int(conv_feat_size), int(conv_feat_size), kernel_size=1, bias=False)
        )

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]
        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )
        '''
        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMRG(
            radii=self.model_cfg.ROI_GRID_POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD
        )
        '''
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = int(mlps[-1][-1])

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict, use_xyz=True):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        '''
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, num_roisx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3) #(B x N x 6x6x6, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        '''
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        conv_features = batch_dict['conv_features']
        xyz = batch_dict["sampled_point"]
        raw_feats = torch.cat([xyz, conv_features], dim=2)
        pooled_feats = self.mlps_(raw_feats) + raw_feats # (B, N, F+3)
        vote_xyz = pooled_feats[:,:,0:3] # (B, N, 3)

        if use_xyz:
            vote_feat = pooled_feats
        else:
            vote_feat = pooled_feats[:,:,3:] # (B, N, F)

        center = rois[:,:,0:3] # (B, num_rois, ,3)
        radius = rois[:,:,3:6] 
        dilate = 1.1
        radius = dilate/2 * radius.square().sum(dim=2).sqrt()# (B, num_rois, 1)
        
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz = vote_xyz.view(-1, 3).contiguous(),
            xyz_batch_cnt = vote_xyz.new_zeros(batch_size).int().fill_(vote_xyz.shape[1]).contiguous(),
            new_xyz  = center.view(-1, 3).contiguous(),
            new_xyz_batch_cnt = center.new_zeros(batch_size).int().fill_(center.shape[1]).contiguous(),
            features = vote_feat.view(-1, vote_feat.size(-1)).contiguous()
        ) # (Bxnum_rois, 3), (Bxnum_rois, C)

        return vote_xyz, pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)

        rcnn_loss_vote_reg, vote_reg_tb_dict = self.get_vote_reg_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_vote_reg
        tb_dict.update(vote_reg_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def get_vote_reg_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        gt_boxes = forward_ret_dict['gt_boxes'] # (B, )
        vote_xyz = forward_ret_dict['vote_xyz'] # (B, N, 3)
        batch_size = forward_ret_dict['batch_size']
        loss = 0

        for bs in range(batch_size):
            vote = vote_xyz[bs, :, :]
            vote = vote.view(-1, 3) # (N, 3)
            gt_box = gt_boxes[bs][:, :7] # (num_gt, 7)
            point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(vote, gt_box) # (num_gt, N)
            mask_cnter = torch.sum(point_masks, dim=0) # (N, )
            fore_point = point_masks[:, mask_cnter!=0] # (num_gt, N1)
            num_fore_point = fore_point.size(1) # number=N1
            _, box_idx = torch.max(fore_point, 0) # (N1, )
            box_xyz = gt_box[box_idx, :3] # (N1, 3)
            vote_ = vote[box_idx, :] # (N1, 3)
            reg_loss = (box_xyz - vote_).abs().sum(dim=1).sum(dim=0) # L1 loss
            reg_loss /= num_fore_point
            loss += reg_loss.item()

        loss = loss * loss_cfgs.LOSS_WEIGHTS['rcnn_vote_reg_weight'] #TODO: 记得设置
        tb_dict = {'rcnn_vote_reg_weight': loss}

        return loss, tb_dict

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        vote_xyz, pooled_features = self.roi_grid_pool(batch_dict)  # (B x num_rois, C)
        pooled_features = pooled_features.unsqueeze(-1) # (B x num_rois, C, 1)
        shared_features = self.shared_fc_layer(pooled_features) # (B x num_rois, C, 1) -> (B x num_rois, 256, 1)

        '''
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1)) 
        #(BxN, 216xC, 1) -> (BxN, 256,1)
        # -> (B x num_rois, 256)
        '''
        # (BxN, 256,1) -> (BxN, num_class, 1) -> (BxN, num_class)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B x cls, 1 or 3)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 7 x num_cls)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['vote_xyz'] = vote_xyz
            targets_dict['gt_boxes'] = batch_dict['gt_boxes']
            targets_dict['batch_size'] = batch_dict['batch_size']

            self.forward_ret_dict = targets_dict

        return batch_dict
