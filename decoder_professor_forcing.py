import tensorflow as tf
import numpy as np

class ProfessorForcingDecoder(tf.keras.layers.Layer):
    def __init__(self, 
                 num_celulas, 
                 token_inicio, 
                 token_fim,
                 tipo_celula = 'lstm',
                 num_dim_embedding = 0,
                 tam_vocab = 0,
                 tamanho_batch = 64,
                 dropout = 0.0,
                 atencao='none',
                 janelas_atencao = 0,
                 dropout_atencao = 0.0,
                 limite_maximo = 200,
                 nome = 'ProfessorForcingDecoder'):
        
        self.num_celulas = num_celulas
        self.tipo_celula = tipo_celula.lower()
        self.tam_batch = tamanho_batch
        self.tam_vocab = tam_vocab
        self.atencao = None
        self.token_inicio = token_inicio
        self.token_fim = token_fim
        self.limite_maximo = limite_maximo

        self.classificador = tf.keras.layers.Dense(tam_vocab, activation='softmax')
        self.classificador_temporal = tf.keras.layers.TimeDistributed(self.classificador)
        self.funcao_perda = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        
        if self.tipo_celula == 'lstm':
            self.decoder = tf.keras.layers.LSTM(num_celulas,
                                                return_sequences=True,
                                                return_state=True,
                                                dropout=dropout)
        
        if self.tipo_celula == 'gru':
            self.decoder = tf.keras.layers.GRU(num_celulas,
                                               return_sequences=True,
                                               return_state=True,
                                               dropout=dropout) 

        if atencao.lower() == 'luong':
            self.atencao = tf.keras.layers.Attention(janelas_atencao, dropout=dropout_atencao)

        elif atencao.lower() == 'bahdanau':
            self.atencao = tf.keras.layers.AdditiveAttention(janelas_atencao, dropout=dropout_atencao)

        if num_dim_embedding > 0:
            self.embedding = tf.keras.layers.Embedding(tam_vocab, 
                                                       num_dim_embedding, 
                                                       mask_zero = True)

        super(ProfessorForcingDecoder, self).__init__(name=nome, trainable=True, dynamic=True)

    def decode(self, x, estado_final_anterior, training = False, mask = None):
        if self.tipo_celula == 'lstm':
            x, estado_oculto, estado_celula = self.decoder(x, 
                                                           estado_final_anterior,
                                                           mask = mask,
                                                           training = False)
            estado_final = [estado_oculto, estado_celula]

        elif self.tipo_celula == 'gru':
            x, estado_final = self.decoder(x, 
                                           estado_final_anterior,
                                           mask = mask, 
                                           training = False)
        return x, estado_final

    def call(self, inputs, saida_decoder, estado_final_anterior, training=False):
        if training:
            x = self.embedding(inputs)
            mascara = self.embedding.compute_mask(inputs)
            """
            if self.atencao:
                contexto_atencao = self.atencao([tf.expand_dims(estado_final_anterior, 1), saida_decoder])
                x = tf.concat([tf.expand_dims(contexto_atencao, 1), x], axis=-1)"""
            saida, _ = self.decode(x, 
                                   estado_final_anterior, 
                                   mask = mascara,
                                   training = True)

            saida = self.classificador_temporal(saida)[:,:-1]

        elif inputs is not None:
            saida = np.zeros((1, self.tam_batch, self.tam_vocab), dtype = np.float32)
            saida[0][0][self.token_inicio] = 1.0
            saida = tf.convert_to_tensor(saida, dtype = tf.dtypes.float32)
            x = tf.convert_to_tensor([[self.token_inicio]] * self.tam_batch) # A entrada inicial tem formato (tam_batch)
                                                                           # e é preenchida com o token de início

            estado_final = estado_final_anterior # O estado oculto/final da rede recorrente

            for _ in range(inputs.shape[1]):
                mascara = self.embedding.compute_mask(x) # Usando máscara para a rede recorrente ignorar os zeros
                x = self.embedding(x) 
                x, estado_final = self.decode(x, 
                                              estado_final, 
                                              mask = mascara,
                                              training = False)
                    
                x = tf.squeeze(x, [1]) # Convertendo a saída da rede recorrente de (tam_batch, 1, num_celulas)
                                       # para (tam_batch, num_celulas)

                x = self.classificador(x) # A saída do classificador tem formato (tam_batch, tam_vocab)
                saida = tf.concat([saida, tf.expand_dims(x, 0)], 0)
                x = tf.math.argmax(x, axis = -1, output_type = tf.dtypes.int64)
                x = tf.expand_dims(x, axis = 1)
            
            saida = tf.transpose(saida[1:-1], [1, 0, 2]) # Removendo a primeira posição que foi criada só para inicializar o tensor

        else:
            saida = np.zeros((1, self.tam_vocab), dtype = np.float32)
            saida[0][self.token_inicio] = 1.0
            saida = tf.convert_to_tensor(saida, dtype = tf.dtypes.float32)
            x = tf.convert_to_tensor([[self.token_inicio]], dtype = tf.dtypes.int64)
            estado_final = estado_final_anterior
            limitador = 0

            while x != tf.constant([self.token_fim], dtype = tf.dtypes.int64) and limitador < 200:
                mascara = self.embedding.compute_mask(x) # Usando máscara para a rede recorrente ignorar os zeros
                x = self.embedding(x) 

                x, estado_final = self.decode(x, 
                                              estado_final, 
                                              mask = mascara,
                                              training = False)
                    
                x = tf.squeeze(x, [1]) # Convertendo a saída da rede recorrente de (tam_batch, 1, num_celulas)
                                       # para (tam_batch, num_celulas)

                x = self.classificador(x) # A saída do classificador tem formato (tam_batch, tam_vocab)
                saida = tf.concat([saida, x], 0)
                x = tf.math.argmax(x, axis = -1, output_type = tf.dtypes.int64)
                x = tf.expand_dims(x, axis = 1)
                limitador += 1
            
            saida = saida[:,1:-1] # Removendo a primeira posição que foi criada só para inicializar o tensor
                                  # e a última que é o token final
        if inputs is not None:
            perda = self.funcao_perda(inputs[:,1:], saida)
            return tf.math.argmax(saida, axis = -1, output_type = tf.dtypes.int64), tf.reduce_mean(perda)
        else:
            return tf.math.argmax(saida, axis = -1, output_type = tf.dtypes.int64), 0.0        