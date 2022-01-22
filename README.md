# 大数据下dataloader的设计

详细的解析看：

知乎： https://zhuanlan.zhihu.com/p/460012052

微信公众号： https://mp.weixin.qq.com/s/7OvXkjx_txVpEYumJVj_AA

csdn: https://blog.csdn.net/weixin_42001089/article/details/122641298?spm=1001.2014.3001.5501


# 开始

```
################# train loader #################
# 单机多卡
if hparams.use_ddp_single_machine:
    print(torch.cuda.device_count())
    train_loader = DataLoader(hparams, train_dataset_path, dataset_idx_path, collate_fn, hparams.local_rank, torch.cuda.device_count())
# 多机多卡
elif hparams.use_ddp_multi_machine:
    train_loader = DataLoader(hparams, train_dataset_path, dataset_idx_path, collate_fn, rank, world_size)
# 单卡
else:
    train_loader = DataLoader(hparams, train_dataset_path, dataset_idx_path, collate_fn, 0, 1)
    
################# train loader #################
# dev: 单个进程，只在一张卡上面运行
valid_dataset = LazyDataset(dev_dataset_path)
valid_loader = DataLoader(hparams, collate_fn=collate_fn, is_train=False, dataset=valid_dataset)


############# train loader usage ###############

train_iter = iter(train_loader)
while True:
    if self.steps == self.total_steps + 1:
                break
    batch, batch_size = next(train_iter)
    ...
    

############# dev loader usage ###############
for batch, batch_size in valid_loader:
    ...


```



