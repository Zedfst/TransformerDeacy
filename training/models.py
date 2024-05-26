from tensorflow.keras import layers
import tensorflow as tf
import keras
  
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads,rate=0.0):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(inputs + attn_output)
        return  out1
    
class DecayFactorLayer(tf.keras.layers.Layer):
    def __init__(self,output_dim):
        self.output_dim=output_dim
        super(DecayFactorLayer, self).__init__()
    def build(self, input_shape):
        self.w = self.add_weight(shape=(1,self.output_dim),
                                 initializer='glorot_uniform',
                                 name='weight',
                                 dtype='float32',
                                 trainable=True)

        self.b = self.add_weight(shape=(input_shape[2],1),
                                 initializer='zero',
                                 name='bias',
                                 dtype='float32',
                                 trainable=True)

    def call(self, inputs,training=None):
        delta=inputs
        delta=tf.exp(-tf.nn.relu(delta@self.w + self.b))
        return delta
    def get_config(self):
        config = super(DecayFactorLayer, self).get_config()
        config.update({"output_dimession": self.output_dim})
        return config

class TransformerDecay(keras.Model):

    def __init__(self,maxlen,vocab_size,nr_visits,embed_dim,
                 num_heads,transformer_dropout_rate,learning_mode='with_decay'):

        self.maxlen=maxlen
        self.vocab_size=vocab_size
        self.nr_visits=nr_visits
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.learning_mode=learning_mode
        self.transformer_dropout_rate=transformer_dropout_rate
        super(TransformerDecay, self).__init__()

        self.embedding_layer = layers.Embedding(input_dim=vocab_size, 
                                                output_dim=embed_dim,mask_zero=True,
                                                embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.02))

        self.decay_factor_layer=DecayFactorLayer(self.embed_dim)
        self.transformer_block=TransformerBlock(self.embed_dim,self.num_heads,self.transformer_dropout_rate)
        
        self.dropout1=layers.Dropout(0.1)
        self.dropout2=layers.Dropout(0.8)
        
        self.dense_demo=layers.Dense(15,activation='relu',name="dense_demo")
        self.classifier=layers.Dense(1,activation='sigmoid')


    def call(self, inputs,training=None):
        #clinical codes
        codes=tf.cast(inputs[0],tf.float32)
        #patient demographics
        demographics=tf.cast(inputs[1],tf.float32)
        #elapsed days
        time=tf.cast(inputs[2][:,1],tf.float32)

        clinical_code_embeddings=self.embedding_layer(codes)
        no_codes=clinical_code_embeddings.shape[1]//2


        #Time weighting
        time=tf.reshape(time,[-1,1,1])
        if self.learning_mode=="with_decay":
            decay_factor=self.decay_factor_layer(time)
            decay_factor=tf.tile(decay_factor,[1,no_codes,1])
            decay_factor=tf.concat([decay_factor,tf.ones_like(decay_factor)],1)
            clinical_code_embeddings*=decay_factor

        #Transformer encoder
        clinical_code_embeddings=self.dropout1(clinical_code_embeddings,training=training)
        clinical_code_embeddings=self.transformer_block(clinical_code_embeddings)
        
        # Sum of clinical codes embeddings of visit 1 and visit 2
        clinical_code_embeddings=tf.reduce_sum(clinical_code_embeddings,1)
        clinical_code_embeddings=self.dropout2(clinical_code_embeddings,training=training)


        #Concatenate clinical code embeddings with patient demographics
        demographics=self.dense_demo(demographics)
        latent_state=tf.concat([clinical_code_embeddings,demographics],1)
        



        #Classifier
        output=self.classifier(latent_state)

        return output
    
    def get_config(self):
        config = super(TransformerDecay, self).get_config()
        config.update({"maxlen": self.maxlen})
        config.update({"vocab_size": self.vocab_size})
        config.update({"embed_dim": self.embed_dim})
        config.update({"num_heads": self.num_heads})
        config.update({"learning_mode": self.learning_mode})
        config.update({"transformer_dropout_rate": self.transformer_dropout_rate})
        return config