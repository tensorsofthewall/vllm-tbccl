import os, sys, torch, torch.distributed as dist, datetime
rank=int(os.environ["RANK"]); init=sys.argv[sys.argv.index("--init")+1]
dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
t=torch.ones(4)*(rank+1); dist.all_reduce(t); print("gloo ok", t.tolist(), flush=True)
