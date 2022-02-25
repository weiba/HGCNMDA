from utils import *
import torch as t
from torch import nn
from torch_geometric.nn import conv
from torch_geometric.nn import dense, norm
import copy

class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.mdg = sizes.mdg
        self.d = sizes.d
        self.m = sizes.m
        self.g = sizes.g

        self.linear_x_4 = nn.Linear(self.d, self.fg)
        self.linear_x_5 = nn.Linear(self.m, self.fg)
        self.linear_x_6 = nn.Linear(self.g, self.fg)

        self.gcn_xdd = conv.GCNConv(self.d, self.fg)
        self.gcn_xmd = conv.GCNConv(self.fg, self.fg)
        self.gcn_xmm = conv.GCNConv(self.m, self.fg)
        self.gcn_xdg = conv.GCNConv(self.fg, self.fg)
        self.gcn_xmg = conv.GCNConv(self.fg, self.fg)

        self.rgcn_d = conv.RGCNConv(self.fg, self.fg, 14)

        self.linear_x_1 = nn.Linear(256, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_x_11 = nn.Linear(256, 256)
        self.linear_x_22 = nn.Linear(256, 128)
        self.linear_x_33 = nn.Linear(128, 64)

        self.linear_x_111 = nn.Linear(256, 256)
        self.linear_x_222 = nn.Linear(256, 128)
        self.linear_x_333 = nn.Linear(128, 64)

    def forward(self, input):
        t.manual_seed(1)
        ##d,m,g
        d_dim = input[0]['data'].size(0)
        m_dim = input[1]['data'].size(0)
        g_dim = input[2]['data'].size(0)

        dm_edge = input[8]['edge_index']
        md_edge = t.cat((input[8]['edge_index'][1, :].reshape(1, -1), input[8]['edge_index'][0, :].reshape(1, -1)))

        x_d = t.relu(self.linear_x_4(input[0]['data'].cuda()))
        x_m = t.relu(self.linear_x_5(input[1]['data'].cuda()))
        x_g = t.relu(self.linear_x_6(input[2]['data'].cuda()))

        d1d3_eg = input[9]['d1d3_edge_index']
        d3d1_eg = t.cat((input[9]['d1d3_edge_index'][1, :].reshape(1, -1), input[9]['d1d3_edge_index'][0, :].reshape(1, -1)))
        d1m1_eg = input[9]['d1m1_edge_index']
        m1d1_eg = t.cat((input[9]['d1m1_edge_index'][1, :].reshape(1, -1), input[9]['d1m1_edge_index'][0, :].reshape(1, -1)))
        d1m2_eg = input[9]['d1m2_edge_index']
        m2d1_eg = t.cat((input[9]['d1m2_edge_index'][1, :].reshape(1, -1), input[9]['d1m2_edge_index'][0, :].reshape(1, -1)))
        g2d1_eg = input[9]['g2d1_edge_index']
        d1g2_eg = t.cat((input[9]['g2d1_edge_index'][1, :].reshape(1, -1), input[9]['g2d1_edge_index'][0, :].reshape(1, -1)))
        d2m1_eg = input[9]['d2m1_edge_index']
        m1d2_eg = t.cat((input[9]['d2m1_edge_index'][1, :].reshape(1, -1), input[9]['d2m1_edge_index'][0, :].reshape(1, -1)))
        m1m3_eg = input[9]['m1m3_edge_index']
        m3m1_eg = t.cat((input[9]['m1m3_edge_index'][1, :].reshape(1, -1), input[9]['m1m3_edge_index'][0, :].reshape(1, -1)))
        g1m1_eg = input[9]['g1m1_edge_index']
        m1g1_eg = t.cat((input[9]['g1m1_edge_index'][1, :].reshape(1, -1), input[9]['g1m1_edge_index'][0, :].reshape(1, -1)))

        d1d3_nu = input[9]['d1d3_edge_index'].size(1)
        d1m1_nu = input[9]['d1m1_edge_index'].size(1)
        d1m2_nu = input[9]['d1m2_edge_index'].size(1)
        g2d1_nu = input[9]['g2d1_edge_index'].size(1)
        d2m1_nu = input[9]['d2m1_edge_index'].size(1)
        m1m3_nu = input[9]['m1m3_edge_index'].size(1)
        g1m1_nu = input[9]['g1m1_edge_index'].size(1)

        d_edge_index = t.cat((d1d3_eg,d3d1_eg,d1m1_eg,m1d1_eg,d1m2_eg,m2d1_eg,g2d1_eg,d1g2_eg,d2m1_eg,m1d2_eg,m1m3_eg,m3m1_eg,g1m1_eg,m1g1_eg), 1)
        d_edge_type = t.cat((t.zeros(1, d1d3_nu), t.ones(1, d1d3_nu),
                             t.full((1, d1m1_nu), 2.0), t.full((1, d1m1_nu), 3.0),
                             t.full((1, d1m2_nu), 4.0), t.full((1, d1m2_nu), 5.0),
                             t.full((1, g2d1_nu), 6.0), t.full((1, g2d1_nu), 7.0),
                             t.full((1, d2m1_nu), 8.0), t.full((1, d2m1_nu), 9.0),
                             t.full((1, m1m3_nu), 10.0), t.full((1, m1m3_nu), 11.0),
                             t.full((1, g1m1_nu), 12.0), t.full((1, g1m1_nu), 13.0)), 1)
        d_edge_type = t.reshape(d_edge_type, [-1, ])
        #d1
        d1 = x_d
        #m1
        m1 = x_m
        #g1
        g1 = x_g
        # dd get the feature of d2
        x_d2_gcn = t.relu(self.gcn_xdd(input[0]['data'].cuda(), input[0]['edge_index'].cuda()))

        # dm get the feature of d3 m3
        d_dm_md_edge = t.cat((dm_edge, md_edge), 1)
        xd_dm = t.cat((x_d, x_m))
        xd_dm_gcn = t.relu(self.gcn_xmd(xd_dm.cuda(), d_dm_md_edge.cuda()))
        x_d3_gcn, x_m3_gcn2 = t.split(xd_dm_gcn.cuda(), (d_dim, m_dim))

        # mm get the feature of m2
        x_m2_gcn = t.relu(self.gcn_xmm(input[1]['data'].cuda(), input[1]['edge_index'].cuda()))

        #gd get the feature of g1
        m_dg_gcn_edge = input[3]['dg_edge_gcn']
        m_gd_gcn_edge = t.cat((input[3]['dg_edge_gcn'][1, :].reshape(1, -1), input[3]['dg_edge_gcn'][0, :].reshape(1, -1)))
        m_dg_gd_edge = t.cat((m_dg_gcn_edge, m_gd_gcn_edge), 1)
        xm_dg = t.cat((x_d, x_g))
        xm_dg_gcn = t.relu(self.gcn_xdg(xm_dg.cuda(), m_dg_gd_edge.cuda()))
        xm_d_gcn1, x_g1_gcn = t.split(xm_dg_gcn.cuda(), (d_dim, g_dim))

        #gm get the feature of g2
        d_mg_gcn_edge = input[4]['mg_edge_gcn']
        d_gm_gcn_edge = t.cat((input[4]['mg_edge_gcn'][1, :].reshape(1, -1), input[4]['mg_edge_gcn'][0, :].reshape(1, -1)))
        d_mg_gm_edge = t.cat((d_mg_gcn_edge, d_gm_gcn_edge), 1)
        xd_mg = t.cat((x_m, x_g))
        xd_mg_gcn = t.relu(self.gcn_xmg(xd_mg.cuda(), d_mg_gm_edge.cuda()))
        xd_m_gcn1, x_g2_gcn = t.split(xd_mg_gcn.cuda(), (m_dim, g_dim))

        xd_dmg = t.cat((d1, x_d2_gcn, x_d3_gcn, m1, x_m2_gcn, x_m3_gcn2,x_g1_gcn,x_g2_gcn))
        Xd = t.relu(self.rgcn_d(xd_dmg.cuda(), d_edge_index.cuda(), d_edge_type.cuda()))

        X_d1_rgcn, X_d2_rgcn, X_d3_rgcn,X_m1_rgcn, X_m2_rgcn, X_m3_rgcn, X_g1_rgcn,X_g2_rgcn = t.split(Xd.cuda(), (d_dim, d_dim,d_dim,m_dim, m_dim, m_dim,g_dim,g_dim))

        Y_dd = t.add(X_d1_rgcn, X_d2_rgcn)
        Y_d = t.add(Y_dd, X_d3_rgcn)
        Y_mm = t.add(X_m1_rgcn, X_m2_rgcn)
        Y_m = t.add(Y_mm, X_m3_rgcn)
        Y_g = t.add(X_g1_rgcn, X_g2_rgcn)

        X_d_L1 = t.relu(self.linear_x_1(Y_d))
        X_m_L1 = t.relu(self.linear_x_11(Y_m))
        X_g_L1 = t.relu(self.linear_x_111(Y_g))

        X_d_L2 = t.relu(self.linear_x_2(X_d_L1))
        X_m_L2 = t.relu(self.linear_x_22(X_m_L1))
        X_g_L2 = t.relu(self.linear_x_222(X_g_L1))

        X_d_L3 = t.relu(self.linear_x_3(X_d_L2))
        X_m_L3 = t.relu(self.linear_x_33(X_m_L2))
        X_g_L3 = t.relu(self.linear_x_333(X_g_L2))

        dm = X_d_L3.mm(X_m_L3.t())
        md = dm.t()
        dg = X_d_L3.mm(X_g_L3.t())
        gd = dg.t()
        mg = X_m_L3.mm(X_g_L3.t())
        gm = mg.t()
        return md, gd, gm

