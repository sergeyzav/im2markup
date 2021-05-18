import argparse, matplotlib.pyplot as plt, logging, sys, os, matplotlib.image as im
from PIL import Image

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate BLEU score')
    parser.add_argument('--gold-path', dest='gold_path',
                        type=str, required=True,
                        help=('Images directory containing the rendered goal images'
                        ))
    parser.add_argument('--pred-path', dest='pred_path',
                        type=str, required=True,
                        help=('Images directory containing the rendered predicted images'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt'
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    app_dir = os.path.join(script_dir, '../..')

    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    gold_path = parameters.gold_path
    pred_path = parameters.pred_path
    assert os.path.exists(gold_path), 'Images directory containing the rendered goal images {} is not found'.format(gold_path)
    assert os.path.exists(pred_path), 'Images directory containing the rendered predicted images {} with goal  is not found'.format(pred_path)

    for filename in os.listdir(pred_path):
        if (os.path.exists('{}/{}'.format(gold_path, filename))):
            image_gold = im.imread('{}/{}'.format(gold_path, filename))
            image_pred = im.imread('{}/{}'.format(pred_path, filename))
            fig, axes = plt.subplots(nrows=2, ncols=1)

            ax = axes.ravel()

            ax[0].imshow(image_gold, aspect='equal')
            ax[0].set_title('gold')
            ax[0].axis('off')

            ax[1].imshow(image_pred, aspect='equal')
            ax[1].axis('off')
            ax[1].set_title('pred')

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')