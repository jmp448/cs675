from main import get_args, train


def param_sweep_ff(args):
    vals = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    for v in vals:
        args.learning_rate = v
        int_v = int(v * 10**6)
        args.log_file = "logs/ff-logs-lr={}e-6.csv".format(int_v)
        args.model_save = "models/ff-lr={}e-6.torch".format(int_v)
        train(args)


def param_sweep_cnn(args):
    vals = [1, 20, 40, 60, 80, 100]
    for v in vals:
        args.cnn_n1_channels = v
        args.log_file = "logs/cnn-logs-nchan={}.csv".format(v)
        args.model_save = "models/cnn-nchan={}.torch".format(v)
        train(args)


def param_sweep_best(args):
    n1_channel_vals = [1, 10, 20, 40, 60, 80, 100]
    for v in n1_channel_vals:
        args.best_n1_channels = v
        args.log_file = "logs/best-n1chan={}.csv".format(v)
        args.model_save = "models/best-n1chan={}.torch".format(v)
        train(args)

    n2_channel_vals = [1, 5, 10, 15, 20, 30, 40]
    for v in n2_channel_vals:
        args.best_n2_channels = v
        args.log_file = "logs/best-n2chan={}.csv".format(v)
        args.model_save = "models/best-n2chan={}.torch".format(v)
        train(args)

    n3_channel_vals = [1, 5, 10, 15, 20, 30, 40]
    for v in n3_channel_vals:
        args.best_n3_channels = v
        args.log_file = "logs/best-n3chan={}.csv".format(v)
        args.model_save = "models/best-n3chan={}.torch".format(v)
        train(args)


if __name__ == "__main__":
    ARGS = get_args()
    if ARGS.model == "simple-ff":
        param_sweep_ff(ARGS)
    elif ARGS.model == "simple-cnn":
        param_sweep_cnn(ARGS)
    elif ARGS.model == "best":
        param_sweep_best(ARGS)
