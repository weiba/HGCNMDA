from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        #self.nums = opt.validation
    def __getitem__(self, index):
        return (self.data_set['dd'], self.data_set['mm'],
                self.data_set['gg'], self.data_set['gd'],
                self.data_set['gm'], self.data_set['md']['train'],
                self.data_set['md_p'], self.data_set['md_true'],
                self.data_set['dm'], self.data_set['RGCN_edge'])

    # def __len__(self):
    #     return self.nums