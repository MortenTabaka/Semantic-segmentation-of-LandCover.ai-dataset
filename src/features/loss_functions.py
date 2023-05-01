import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import BinaryCrossentropy, binary_crossentropy

from src.features.data_features import MaskFeatures


class SemanticSegmentationLoss(object):
    """
    Loss functions for semantic segmentation:
        Binary Cross Entropy
        Weighted Cross Entropy
        Balanced Cross Entropy
        Dice Loss
        Focal loss
        Tversky loss
        Focal Tversky loss
        log-cosh dice loss

    All credits go to (not including soft_dice_loss) Shruti Jadon.

    SemSegLoss: A python package of loss functions for semantic segmentation, 2021.
    https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions

    @inproceedings{jadon2020survey,
        title={A survey of loss functions for semantic segmentation},
        author={Jadon, Shruti},
        booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
        pages={1--7},
        year={2020},
        organization={IEEE}
    }
    @article{JADON2021100078,
        title = {SemSegLoss: A python package of loss functions for semantic segmentation},
        journal = {Software Impacts},
        volume = {9},
        pages = {100078},
        year = {2021},
        issn = {2665-9638},
        doi = {https://doi.org/10.1016/j.simpa.2021.100078},
        url = {https://www.sciencedirect.com/science/article/pii/S2665963821000269},
        author = {Shruti Jadon},
        keywords = {Deep Learning, Image segmentation, Medical imaging, Loss functions},
        abstract = {Image Segmentation has been an active field of research as it has a wide range of applications, ranging from automated disease detection to self-driving cars. In recent years, various research papers proposed different loss functions used in case of biased data, sparse segmentation, and unbalanced dataset. In this paper, we introduce SemSegLoss, a python package consisting of some of the well-known loss functions widely used for image segmentation. It is developed with the intent to help researchers in the development of novel loss functions and perform an extensive set of experiments on model architectures for various applications. The ease-of-use and flexibility of the presented package have allowed reducing the development time and increased evaluation strategies of machine learning models for semantic segmentation. Furthermore, different applications that use image segmentation can use SemSegLoss because of the generality of its functions. This wide range of applications will lead to the development and growth of AI across all industries.}
    }
    """

    def __init__(
        self,
        number_of_classes: int,
        beta: float = 0.25,
        alpha: float = 0.25,
        gamma: float = 2,
        epsilon: float = 1e-6,
        smooth: float = 1,
    ):
        self.number_of_classes = number_of_classes
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + K.epsilon()) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()
        )

    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def convert_to_logits(self, y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropy_loss(self, y_true, y_pred):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = self.beta / (1 - self.beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=y_pred, targets=y_true, pos_weight=pos_weight
        )
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
            weight_a + weight_b
        ) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(
            logits=logits,
            targets=y_true,
            alpha=self.alpha,
            gamma=self.gamma,
            y_pred=y_pred,
        )

        return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        sigmoided_matrix = self.sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2.0 * intersection + self.smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth
        )
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        y_pred_pos = K.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = K.sum(y_pos * y_pred_pos)
        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        prec = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = (K.sum(y_pos * y_pred_pos) + self.smooth) / (K.sum(y_pos) + self.smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (K.sum(y_neg * y_pred_neg) + self.smooth) / (K.sum(y_neg) + self.smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + self.smooth) / (
            true_pos + alpha * false_neg + (1 - alpha) * false_pos + self.smooth
        )

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return K.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

    def jacard_loss(self, y_true, y_pred):
        """
        Intersection-Over-Union (IoU), also known as the Jaccard loss
        """
        return 1 - self.jacard_similarity(y_true, y_pred)

    def unet3p_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
        Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
        which is able to capture both large-scale and fine structures with clear boundaries.
        """
        focal_loss = self.focal_loss(y_true, y_pred)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)

        return focal_loss + ms_ssim_loss + jacard_loss

    def basnet_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
        The hybrid loss is a combination of the binary cross entropy, structural similarity
        and intersection-over-union losses, which guide the network to learn
        three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
        """
        bce_loss = BinaryCrossentropy(from_logits=False)
        bce_loss = bce_loss(y_true, y_pred)

        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)

        return bce_loss + ms_ssim_loss + jacard_loss

    def soft_dice_loss(self, y_true, y_pred):
        """
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
        Source: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
            epsilon: Used for numerical stability to avoid divide by zero errors

        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        """
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape) - 1))
        numerator = 2.0 * tf.reduce_sum(y_pred * y_true, axes)
        denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), axes)

        return 1 - tf.reduce_mean(
            (numerator + self.epsilon) / (denominator + self.epsilon)
        )  # average over classes and batch

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + K.exp(-x))

    @staticmethod
    def jacard_similarity(y_true, y_pred):
        """
        Intersection-Over-Union (IoU), also known as the Jaccard Index
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

    @staticmethod
    def ssim_loss(y_true, y_pred):
        """
        Structural Similarity Index (SSIM) loss
        """
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)
