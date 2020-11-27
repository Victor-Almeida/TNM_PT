# Papinemi et al., 2002 (https://dl.acm.org/doi/10.3115/1073083.1073135)
# OBS : executar em um loop customizado fora do escopo @tf.function

import numpy as np

def calcular_bleu(tensor_referencias, tensor_candidatos, ngrams = 4):
    referencias = tensor_referencias.numpy()
    candidatos = tensor_candidatos.numpy()
    batch_bleu = 0
    peso = 1 / ngrams
    
    for index in range(len(referencias)):
        bleu = 0

        for ngram in range(1, ngrams + 1):
            referencia = montar_ngram(referencias[index], ngram = ngram)
            candidato = montar_ngram(candidatos[index], ngram = ngram)
            count = 0

            for gram_candidato in montar_conjunto(candidato):
                count += referencia.count(gram_candidato)

            if len(montar_conjunto(candidato)) > 0 and count > 0:
                count = count / len(montar_conjunto(candidato))
                bleu += (peso * np.log(count))

        r = len(montar_conjunto(referencias[index]))
        c = len(montar_conjunto(candidatos[index]))

        if c > 0:
            bp = 1 if c > r else np.exp((1 - r)/c)
            bleu = bp * np.exp(bleu)
            batch_bleu += bleu 
        else:
            pass              

    return batch_bleu/len(referencias)

def montar_ngram(vetor, ngram = 2, remover_zeros = True):
    novo_vetor = []
    vetor_ngram = []

    for elemento in vetor:
        if elemento != 0:
            novo_vetor.append(elemento)
    
    for i in range(len(novo_vetor) - ngram + 1 ):
        vetor_ngram.append(tuple(novo_vetor[i:i+ngram]))

    return vetor_ngram

def montar_conjunto(vetor):
    conjunto = set()

    for elemento in vetor:
        if elemento not in conjunto:
            conjunto.add(elemento)

    return conjunto