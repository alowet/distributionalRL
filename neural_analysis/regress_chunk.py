import json
from regress_session import regress_session

def regress_chunk(chunk, table='ephys', refit=False, lr=5e-3, reg='group_lasso', se_frac=0.75, l1=0.9):
    print(chunk)
    sessions = json.loads(chunk)
    for session in sessions:
        print(session)
        regress_session(*session, table=table, refit=refit, lr=lr, reg=reg, se_frac=se_frac, l1_ratio=l1)
        # regress_session(*session)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Regress chunk')
    parser.add_argument('chunk')
    parser.add_argument('-t', '--table', default='ephys')
    parser.add_argument('-r', '--refit', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=5e-3)
    parser.add_argument('-g', '--regularization', default='group_lasso')  # or 'elastic_net'
    parser.add_argument('-f', '--se_frac', type=float, default=.75)
    parser.add_argument('-i', '--l1_ratio', type=float, default=0.9)
    args = parser.parse_args()
    regress_chunk(args.chunk, table=args.table, refit=args.refit, lr=args.lr, reg=args.regularization, se_frac=args.se_frac, l1=args.l1_ratio)
    # regress_chunk(args.chunk)

