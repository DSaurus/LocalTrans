def parse_config(argv=None):
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='localtrans')
    cfg = {}

    cfg["div_scale_list"] = [16, 8, 4]
    cfg["kernel_list"] = [9, 7, 5]
    cfg["bias_list"] = [0.5, 0.2, 0.075, 0.02]
    cfg["image_size"] = (128, 128)

    # general settings                              
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str, default='localtrans', help='name of a model/experiment.')
    parser.add_argument('--dataroot', type=str, default='train2014', help='name of a model/experiment.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--random_noise', type=bool, default=True)
    parser.add_argument('--random_color', type=bool, default=True)
    parser.add_argument('--random_bias', type=float, default=0.5)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--resume_dir', type=str)

    args, _ = parser.parse_known_args()

    return args