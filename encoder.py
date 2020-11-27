import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, 
                 num_celulas, 
                 tipo_celula='lstm',
                 num_dim_embedding = 0,
                 tam_vocab = 0,
                 dropout = 0.0,
                 nome='Encoder'):
        
        self.num_celulas = num_celulas
        self.tipo_celula = tipo_celula.lower()
        
        if self.tipo_celula == 'lstm':
            self.encoder = tf.keras.layers.LSTM(num_celulas,
                                                return_sequences=True,
                                                return_state=True,
                                                dropout=dropout)
        
        if self.tipo_celula == 'gru':
            self.encoder = tf.keras.layers.GRU(num_celulas,
                                               return_sequences=True,
                                               return_state=True,
                                               dropout=dropout) 
            
        if num_dim_embedding > 0:
            self.embedding = tf.keras.layers.Embedding(tam_vocab, 
                                                       num_dim_embedding, 
                                                       mask_zero = True)

        super(Encoder, self).__init__(name=nome, trainable=True, dynamic=True)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        mascara = self.embedding.compute_mask(inputs)

        if self.tipo_celula == 'lstm':
            saida, estado_oculto, estado_celula = self.encoder(x, 
                                                               mask = mascara,
                                                               training = training)
            estado_final = [estado_oculto, estado_celula]
        elif self.tipo_celula == 'gru':
            saida, estado_final = self.encoder(x, 
                                               mask = mascara,
                                               training=training)

        return saida, estado_final