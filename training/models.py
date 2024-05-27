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
    
    
class GRUDECAYLAYERS(keras.layers.Layer):
    def __init__(self, unit_1,dropout_grud,**kwargs):
        self.unit_1 = unit_1
        self.dropout_grud=dropout_grud
        self.dropout=layers.Dropout(self.dropout_grud)
        self.state_size = [tf.TensorShape([unit_1])]
        super(GRUDECAYLAYERS, self).__init__(**kwargs)

    def build(self, input_shapes):
        i1 = input_shapes[1]-1

        
        self.w_z = self.add_weight(
            shape=(i1, self.unit_1), initializer="glorot_uniform", name="w_z",trainable=True
        )
        self.u_z = self.add_weight(
            shape=(self.unit_1, self.unit_1), initializer="glorot_uniform", name="u_z",trainable=True
        )
        self.b_z = self.add_weight(
            shape=(1,), initializer="zero", name="b_z",trainable=True
        )
        
        
        
        self.w_h = self.add_weight(
            shape=(i1, self.unit_1), initializer="glorot_uniform", name="w_h",trainable=True
        )
        self.u_h = self.add_weight(
            shape=(self.unit_1, self.unit_1), initializer="glorot_uniform", name="u_h",trainable=True
        )
        self.b_h = self.add_weight(
            shape=(1,), initializer="zero", name="b_h",trainable=True
        )
        
        
        self.w_r = self.add_weight(
            shape=(i1, self.unit_1), initializer="glorot_uniform", name="w_r",trainable=True
        )
        self.u_r = self.add_weight(
            shape=(self.unit_1, self.unit_1), initializer="glorot_uniform", name="u_r",trainable=True
        )
        self.b_r = self.add_weight(
            shape=(1,), initializer="zero", name="b_r",trainable=True
        )
        

        
        self.w_int = self.add_weight(
            shape=(1, self.unit_1), initializer="glorot_uniform", name="w_int",constraint=tf.keras.constraints.NonNeg(),trainable=True
        )
        self.b_int = self.add_weight(
            shape=(1,), initializer="zero", name="b_int",constraint=tf.keras.constraints.NonNeg(),trainable=True
        )
        

    def call(self, inputs, states,training=None):
        axis=(inputs.shape[1]-1)
        x=inputs[:,:axis]


        interval=tf.reshape(inputs[:,axis:],[-1,1])
        h_=states

        latent_interval=tf.exp(-tf.nn.relu(interval*self.w_int+self.b_int))

        z=tf.nn.sigmoid(tf.matmul(x,self.w_z)+tf.matmul(h_,self.u_z)+self.b_z)
        r=tf.nn.sigmoid(tf.matmul(x,self.w_r)+tf.matmul(h_,self.u_r)+self.b_r)
        h_hat=tf.nn.tanh(tf.matmul(x,self.w_h)+tf.matmul(h_*r,self.u_h)+self.b_h)
        h=tf.squeeze(z*h_+(1-z)*h_hat,0)
        h=latent_interval*h
        
        if training:
            h=self.dropout(h)

        
        return h,[h]
    
    def get_config(self):
        config = super(GRUDECAYLAYERS, self).get_config()
        config.update({"gru_decay_units": self.unit_1})
        config.update({"dropout_gru_decay": self.dropout_grud})
        return config
    
class GRUDecay(keras.Model):
    
    def __init__(self,
                 embedding_input_dim,
                 embedding_output_dim,
                 embedding_input_length,
                 dropout_grud=0.0,
                 dropout_demo=0.1,
                 units=50,
                 units_dense1=50):
        
        super(GRUDecay, self).__init__()
        
        self.embedding_input_dim=embedding_input_dim
        self.embedding_output_dim=embedding_output_dim
        self.embedding_input_length=embedding_input_length
        self.dropout_grud=dropout_grud
        self.dropout_demo=dropout_demo
        
        self.embedding=layers.Embedding(self.embedding_input_dim,
                                                  self.embedding_output_dim,
                                                  input_length=self.embedding_input_length,
                                                  mask_zero=True,
                                                  embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.03),
                                                  name="embedding")
  
      
        self.flatten=layers.Flatten()
        self.gru_decay_layer=GRUDECAYLAYERS(units,self.dropout_grud)
        self.gru_d=tf.keras.layers.RNN(self.gru_decay_layer)
        self.dense_demo=layers.Dense(15,activation="relu",name="dense_demo")
        self.dense1=layers.Dense(units_dense1,activation="leaky_relu",name="dense1")
        self.dense2=layers.Dense(100,activation="leaky_relu",name="dense2")
        self.dropout1=layers.Dropout(self.dropout_demo)
        self.classifier=layers.Dense(1,activation="sigmoid", name="classifier")
        
        self.lstm=layers.LSTM(80,dropout=0.5)
       

    def call(self, inputs,training=None):
        
        
        grouped_codes=inputs[0]
        time=tf.cast(inputs[1],tf.float32)
        demos=tf.cast(inputs[2],tf.float32)
        
        
        grouped_codes=self.embedding(grouped_codes)
        
        axis=grouped_codes.shape[1]//2
        codes_0=grouped_codes[:,:axis,:]
        codes_1=grouped_codes[:,axis:axis*2,:]

        
        
        codes_0=tf.reduce_mean(codes_0,1)
        codes_1=tf.reduce_mean(codes_1,1)


        codes_0=tf.reshape(codes_0,[-1,1,codes_0.shape[1]])
        codes_1=tf.reshape(codes_1,[-1,1,codes_1.shape[1]])

        
        
        codes_0=self.flatten(codes_0)
        codes_0=tf.concat([codes_0,tf.reshape(time[:,0],[-1,1])],1)
        codes_0=tf.reshape(codes_0,[-1,1,codes_0.shape[1]])
        codes_1=self.flatten(codes_1)
        codes_1=tf.concat([codes_1,tf.reshape(time[:,1],[-1,1])],1)
        codes_1=tf.reshape(codes_1,[-1,1,codes_1.shape[1]])
        

        codes=tf.concat([codes_0,codes_1],1)
        codes=self.gru_d(codes)
        codes=self.dense1(codes) 
        
        
        demos=self.dense_demo(demos)
        codes=tf.concat([codes,demos],1)
        codes=self.dense2(codes) 
        
        codes=self.classifier(codes)
        return codes
    
    def get_config(self):
        config = super(GRUDecay, self).get_config()
        config.update({"embedding_input_dim": self.embedding_input_dim})
        config.update({"embedding_output_dim": self.embedding_output_dim})
        config.update({"embedding_input_length": self.embedding_input_length})
        config.update({"dropout_grud": self.dropout_grud})

        return config
    
class ATTENTIONS(keras.layers.Layer):
    def __init__(self,dropout_val=0.0,**kwargs):
        self.dropout_val=dropout_val
        self.dropout=layers.Dropout(self.dropout_val)
        super(ATTENTIONS, self).__init__(**kwargs)

    def build(self, input_shapes):
        axis_1=input_shapes[0][1]
        axis_2=input_shapes[0][2]
        self.w_g = self.add_weight(shape=(axis_2,1),
                                 initializer='glorot_uniform',
                                 name='wg',
                                 dtype='float32',
                                 trainable=True)
        self.b_g = self.add_weight(shape=(axis_1,),
                                 initializer='zero',
                                 name='bg',
                                 dtype='float32',
                                 trainable=True)
        self.w_h = self.add_weight(shape=(axis_2,1),
                                 initializer='glorot_uniform',
                                 name='hg',
                                 dtype='float32',
                                 trainable=True)
        self.b_h = self.add_weight(shape=(axis_1,),
                                 initializer='zero',
                                 name='hg',
                                 dtype='float32',
                                 trainable=True)                         

    def call(self, inputs,training=None):
        v_g=inputs[0]
        v_g=tf.nn.softmax(tf.squeeze(v_g@self.w_g,2)+self.b_g)
        
        v_h=inputs[1]
        v_h=tf.nn.tanh(tf.squeeze(v_h@self.w_h,2)+self.b_h)
            
        return v_g*v_h
    
    def get_config(self):
        config = super(ATTENTIONS, self).get_config()
        return config
    
class RETAIN(keras.Model):
    
    def __init__(self,
                 embedding_input_dim,
                 embedding_output_dim,
                 embedding_input_length,
                 dropout_value=0.0,
                 gru_units=50,):
        
        super(RETAIN, self).__init__()
        
        self.embedding_input_dim=embedding_input_dim
        self.embedding_output_dim=embedding_output_dim
        self.embedding_input_length=embedding_input_length
        self.dropout_value=dropout_value
        self.gru_units=gru_units
        
        self.embedding=layers.Embedding(self.embedding_input_dim,
                                        self.embedding_output_dim,
                                        input_length=self.embedding_input_length,
                                        mask_zero=True,
                                        embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.03),
                                        name="embedding")
        
        self.rnng=layers.LSTM(self.gru_units,dropout=self.dropout_value,return_sequences=True)
        self.rnnh=layers.LSTM(self.gru_units,dropout=self.dropout_value,return_sequences=True)
        self.flatten=layers.Flatten()
        self.attention=ATTENTIONS()
        self.dense_demo=layers.Dense(15,activation="relu",name="dense_demo")
        self.dense2=layers.Dense(80,activation="relu",name="dense2")
        self.classifier=layers.Dense(1,activation="sigmoid", name="classifier")
        
        self.dropout1=layers.Dropout(0.1)
        
    def call(self, inputs,training=None):
        grouped_codes=inputs[0]
        grouped_codes=self.embedding(grouped_codes)
        axis=grouped_codes.shape[1]//2
        codes_0=grouped_codes[:,:axis,:]
        codes_1=grouped_codes[:,axis:axis*2,:]

        
        codes_0=tf.reduce_sum(codes_0,1)
        codes_1=tf.reduce_sum(codes_1,1)

        
        codes_0=tf.reshape(codes_0,[-1,1,codes_0.shape[1]])
        codes_1=tf.reshape(codes_1,[-1,1,codes_1.shape[1]])

        codes=tf.concat([codes_0,codes_1],1)
        
        v_g=self.rnng(tf.reverse(codes,[1]))
        v_h=self.rnnh(tf.reverse(codes,[1]))
        
        coef=tf.reshape(self.attention([v_g,v_h]),[-1,codes.shape[1],1])
        codes=self.dense2(tf.reduce_sum(codes*coef,1))
        
        demos=tf.cast(inputs[1],tf.float32)

        
        codes=tf.concat([codes,demos],1)

        
        codes=self.classifier(codes)
        return codes
    
class IRLLAYERS(tf.keras.layers.Layer):
    
    def __init__(self,linear_units):
        self.linear_units=linear_units
        self.dropout=layers.Dropout(0.0)
        super(IRLLAYERS, self).__init__()

    def build(self, input_shape):
        axis_1=input_shape[0][1]
        axis_2=input_shape[0][2]
        
        self.linear_query = self.add_weight(shape=(axis_2,self.linear_units),
                                 initializer='glorot_uniform',
                                 name='lin_query',
                                 dtype='float32',
                                 trainable=True)
        self.linear_key = self.add_weight(shape=(axis_2,self.linear_units),
                                 initializer='glorot_uniform',
                                 name='lin_key',
                                 dtype='float32',
                                 trainable=True)
        
        self.theta_cn = self.add_weight(shape=(1,axis_1,),
                                 initializer='glorot_uniform',
                                 name='theta_cn',
                                 dtype='float32',
                                 trainable=True)
        self.mu_cn = self.add_weight(shape=(1,axis_1,),
                                 initializer='glorot_uniform',
                                 name='mu_cn',
                                 dtype='float32',
                                 trainable=True)
        
    def call(self, inputs,training=None):
        query=inputs[0]@self.linear_query
        key=inputs[0]@self.linear_key
        scla_fact=tf.sqrt(tf.cast(self.linear_units,tf.float32))
        
        if training:
            query=self.dropout(query)
            key=self.dropout(key)

        alphas=tf.nn.softmax((query@tf.transpose(key,[0,2,1]))/scla_fact)
        gn=tf.reshape(alphas,[-1,alphas.shape[1],alphas.shape[2],1])
        input_=tf.reshape(inputs[0],[-1,inputs[0].shape[1],1,inputs[0].shape[2]])        
        gn=tf.reduce_sum(gn*input_,2)        
        time_fact=tf.nn.sigmoid(self.theta_cn-self.mu_cn*inputs[1])
        if training:
            time_fact=self.dropout(time_fact)
        time_fact=tf.reshape(time_fact,[-1,time_fact.shape[1],1])
        v=gn*time_fact
        v=tf.reduce_sum(v,1)

        return v
    
    def get_config(self):
        config = super(IRLLAYERS, self).get_config()
        return config
    


class Timeline(keras.Model):
    
    def __init__(self,embedding_input_dim,
                 embedding_output_dim,
                 embedding_input_length,):
        
        self.embedding_input_dim=embedding_input_dim
        self.embedding_output_dim=embedding_output_dim
        self.embedding_input_length=embedding_input_length
        super(Timeline, self).__init__()
        self.embedding=layers.Embedding(self.embedding_input_dim,
                                          self.embedding_output_dim,
                                          input_length=self.embedding_input_length,
                                          mask_zero=True,
                                          embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.03),
                                          name="embedding")
        
        self.irl=IRLLAYERS(50)
        self.bilstm=layers.Bidirectional(layers.LSTM(80,dropout=0.8))
        self.dense_demo=layers.Dense(15,activation="relu",name="dense_demo")
        self.dense1=layers.Dense(100,activation="relu",name="dense1")
        self.classifier=layers.Dense(1,activation="sigmoid", name="classifier")
        self.dropout1=layers.Dropout(0.1)

        
    def call(self, inputs,training=None):
        
        codes=inputs[0]
        demos=inputs[2]
        demos=self.dense_demo(demos)
        demos=self.dropout1(demos,training=training)
        
        idays=tf.cast(inputs[1],tf.float32)
        codes=self.embedding(codes)
        axis=codes.shape[1]//2
        codes_0=codes[:,:axis,:]
        codes_1=codes[:,axis:axis*2,:]

        
        v_0=self.irl([codes_0,tf.reshape(idays[:,0],[-1,1])])
        v_1=self.irl([codes_1,tf.reshape(idays[:,1],[-1,1])])

        
        v_0=tf.reshape(v_0,[-1,1,v_0.shape[1]])
        v_1=tf.reshape(v_1,[-1,1,v_1.shape[1]])

        
        
        v=tf.concat([v_0,v_1],1)
        v=self.bilstm(v)
        v=tf.concat([v,demos],1)
        v=self.dense1(v)

        
        v=self.classifier(v)

        return v
    
    def get_config(self):
        config = super(Timeline, self).get_config()
        return config   
#RP stands for readmission prediction
class LSTM_(keras.Model):
    
    def __init__(self,
                 embedding_input_dim,
                 embedding_output_dim,
                 embedding_input_length,
                 lstm_dropout,
                 dropout_value=0.5,):
        
        super(LSTM_, self).__init__()
        
        self.embedding_input_dim=embedding_input_dim
        self.embedding_output_dim=embedding_output_dim
        self.embedding_input_length=embedding_input_length
        self.lstm_dropout=lstm_dropout
        self.dropout_value=dropout_value
        
        self.embedding=layers.Embedding(self.embedding_input_dim,
                                                  self.embedding_output_dim,
                                                  input_length=self.embedding_input_length,
                                                  mask_zero=True,
                                                  embeddings_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.01),
                                                  name="embedding")
  
        self.flatten=layers.Flatten()
        self.dropout_emb=layers.Dropout(0.0)
        self.lstm=layers.LSTM(80,dropout=self.lstm_dropout)
        self.dense_demo=layers.Dense(15,activation="relu",name="dense_demo")
#         self.dense1=layers.Dense(200,activation="relu",name="dense1")
#         self.dense2=layers.Dense(200,activation="relu",name="dense2")
        self.classifier=layers.Dense(1,activation="sigmoid", name="classifier")
        self.dropout1=layers.Dropout(0.1)
                

    def call(self, inputs,training=None):
        
        
        grouped_codes=inputs[0]
        demos=tf.cast(inputs[1],tf.float32)
        grouped_codes=self.embedding(grouped_codes)
        axis=grouped_codes.shape[1]//2
        codes_0=grouped_codes[:,:axis,:]
        codes_1=grouped_codes[:,axis:axis*2,:]

        codes_0=tf.reduce_sum(codes_0,1)
        codes_1=tf.reduce_sum(codes_1,1)

        
        codes_0=tf.reshape(codes_0,[-1,1,codes_0.shape[1]])
        codes_1=tf.reshape(codes_1,[-1,1,codes_1.shape[1]])

        
        
        
        codes=tf.concat([codes_0,codes_1],1)
        codes=self.lstm(codes)
        
        demos=self.dense_demo(demos)
        if training:
            demos=self.dropout1(demos)
        codes=tf.concat([codes,demos],1)

        

        
        codes=self.classifier(codes)
        return codes
    
    def get_config(self):
        config = super(LSTM_, self).get_config()
        config.update({"embedding_input_dim": self.embedding_input_dim})
        config.update({"embedding_output_dim": self.embedding_output_dim})
        config.update({"embedding_input_length": self.embedding_input_length})
        config.update({"lstm_dropout": self.lstm_dropout})
        return config

       
