from hummingbird.ml import convert
import tensorflow as tf

#model = convert(forest, 'torch', X, extra_config={"tree_implementation":"gemm"})
#torch_model = model.model

class GEMMDecisionTreeImpl(tf.Module):

    def __init__(self, skl_model):
        self.container = convert(skl_model, 'torch', extra_config={"tree_implementation":"gemm"})
        self.op = self.container.model._operators[0]

    @tf.function
    def __call__(self, x, train=False):
        x = tf.transpose(x)
        x = tf.cast(x, dtype=tf.float32)
        
        decision_cond = tf.math.less_equal
        if self.op.decision_cond.__name__ == 'le':
            decision_cond = tf.math.less_equal
        elif self.op.decision_cond.__name__ == 'ge':
            decision_cond = tf.math.greater_equal
        elif self.op.decision_cond.__name__ == 'lt':
            decision_cond = tf.math.less
        elif self.op.decision_cond.__name__ == 'gt':
            decision_cond = tf.math.greater
        elif self.op.decision_cond.__name__ == 'eq':
            decision_cond = tf.math.equal
        else:
            decision_cond = tf.math.not_equal
        
        x = decision_cond(tf.linalg.matmul(self.op.weight_1.detach().numpy(), x), self.op.bias_1.detach().numpy())
        x = tf.reshape(x, (self.op.n_trees, self.op.hidden_one_size, -1))

        x = tf.cast(x, dtype=tf.float32)

        x = tf.linalg.matmul(self.op.weight_2.detach().numpy(), x)

        x = tf.reshape(x, (self.op.n_trees * self.op.hidden_two_size, -1)) == self.op.bias_2.detach().numpy()
        x = tf.reshape(x, (self.op.n_trees, self.op.hidden_two_size, -1))

        x = tf.cast(x, dtype=tf.float32)

        x = tf.linalg.matmul(self.op.weight_3.detach().numpy(), x)
        x = tf.reshape(x, (self.op.n_trees, self.op.hidden_three_size, -1))

        x = tf.transpose(tf.reduce_sum(x, 0))

        return tf.math.argmax(x, axis=1), x

'''
X = tf.constant([1,2,3,4,5,6,7,8], shape=[1, 8])

GEMM_model = GEMMDecisionTreeImpl(forest)

y_pred, y = GEMM_model(X)
print(y_pred, y)
'''