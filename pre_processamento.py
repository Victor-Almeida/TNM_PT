import re
import unicodedata
import numpy as np
from tqdm import tqdm
from collections import Counter

def pre_processar(base_entrada, base_saida, max_vocabulos = 50, n_vocabulos_mais_comuns = 50000):
    total_linhas_removidas = 0

    print("Removendo linhas que contém dígitos : \n")

    i = 0
    linhas_removidas = 0
    pbar = tqdm(total = len(base_saida))
    digitos = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    while i < len(base_saida):
        pular = False

        for caractere in base_saida[i]:
            if caractere in digitos:
                del(base_saida[i])
                del(base_entrada[i])
                pular = True
                break

        if not pular:
            for caractere in base_entrada[i]:
                if caractere in digitos:
                    del(base_saida[i])
                    del(base_entrada[i])
                    pular = True
                    break

        if pular:
            linhas_removidas += 1
        else:
            i += 1

        pbar.update(1)
    pbar.close()

    print("\n" + str(linhas_removidas) + " linhas com dígitos removidas")
    total_linhas_removidas += linhas_removidas

    print("\nRemovendo caracteres especiais : ")
    # Adaptado de : https://www.tensorflow.org/tutorials/text/nmt_with_attention#download_and_prepare_the_dataset

    pbar = tqdm(total = len(base_saida))
    counter_saida = Counter()
    counter_entrada = Counter()

    for i in range(len(base_saida)):
        base_saida[i] = ''.join(caractere for caractere in unicodedata.normalize('NFD', base_saida[i]) if unicodedata.category(caractere) != 'Mn')
        base_entrada[i] = ''.join(caractere for caractere in unicodedata.normalize('NFD', base_entrada[i]) if unicodedata.category(caractere) != 'Mn')

        # Removendo formatação e convertendo os caracteres para minúsculo
        base_saida[i] = base_saida[i].strip().lower()
        base_entrada[i] = base_entrada[i].strip().lower()
        
        # Adicionando um espaço entre uma palavra e a pontuação seguinte
        # ex: "he is a boy." => "he is a boy ."
        # Referência: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        base_saida[i] = re.sub(r"([?.!,;])", r" \1 ", base_saida[i])
        base_entrada[i] = re.sub(r"([?.!,;])", r" \1 ", base_entrada[i])

        base_saida[i] = re.sub(r'[" "]+', " ", base_saida[i])
        base_entrada[i] = re.sub(r'[" "]+', " ", base_entrada[i])

        # Substituir tudo por espaço, exceto (a-z, A-Z, ".", "?", "!", ",")
        base_saida[i] = re.sub(r"[^a-zA-Z?.!,]+", " ", base_saida[i])
        base_entrada[i] = re.sub(r"[^a-zA-Z?.!,]+", " ", base_entrada[i])

        base_saida[i] = base_saida[i].strip()
        base_entrada[i] = base_entrada[i].strip()

        # Adicionando tokens de início e fim de sentença
        # para que o modelo saiba quando começar e terminar a predição
        base_saida[i] = '<inicio> ' + base_saida[i] + ' <fim>'
        base_entrada[i] = '<inicio> ' + base_entrada[i] + ' <fim>'

        # Tokenizando
        base_saida[i] = base_saida[i].split(' ')
        base_entrada[i] = base_entrada[i].split(' ')

        # Contando as ocorrências de cada vocábulo
        counter_saida.update(base_saida[i])
        counter_entrada.update(base_entrada[i])

        pbar.update(1)
    pbar.close()

    print("\nRemovendo linhas com mais de " + str(max_vocabulos) + " vocábulos : ")

    pbar = tqdm(total = len(base_saida))
    n_linhas_removidas = 0
    index = 0

    while index < len(base_saida):
        if len(base_saida[index]) > 50 or len(base_entrada[index]) > 50:
            n_linhas_removidas += 1
            del(base_saida[index])
            del(base_entrada[index])
        else:
            index += 1
        pbar.update(1)
    pbar.close()

    print("\n" + str(n_linhas_removidas) + " linhas com mais de " + str(max_vocabulos) + " vocábulos removidas\n")

    print("Construindo o vocabulário com os " + str(n_vocabulos_mais_comuns) + " vocábulos mais comuns")

    mais_frequentes_saida = [vocabulo for (vocabulo, cont) in counter_saida.most_common(n_vocabulos_mais_comuns)]
    mais_frequentes_entrada = [vocabulo for (vocabulo, cont) in counter_entrada.most_common(n_vocabulos_mais_comuns)]

    dict_saida = dict()
    dict_entrada = dict()

    dict_saida = ({'': 0, '<desconhecido>': 1})
    dict_saida[0] = ''
    dict_saida[1] = '<desconhecido>'

    dict_entrada = ({'': 0, '<desconhecido>': 1})
    dict_entrada[0] = ''
    dict_entrada[1] = '<desconhecido>'

    for i in range(n_vocabulos_mais_comuns):
        dict_saida[i + 2] = mais_frequentes_saida[i]
        dict_saida[mais_frequentes_saida[i]] = i + 2

        dict_entrada[i + 2] = mais_frequentes_entrada[i]
        dict_entrada[mais_frequentes_entrada[i]] = i + 2

    print("\nTokenizando e preenchendo linhas menores do que " + str(max_vocabulos) + " com zeros : ")

    pbar = tqdm(total = len(base_saida))

    for i in range(len(base_saida)):
        for index_token in range(len(base_saida[i])):
            if base_saida[i][index_token] in dict_saida:
                base_saida[i][index_token] = dict_saida[base_saida[i][index_token]]
            else:
                base_saida[i][index_token] = dict_saida['<desconhecido>']
        base_saida[i] += [0] * (50 - len(base_saida[i]))
        
        
        for index_token in range(len(base_entrada[i])):
            if base_entrada[i][index_token] in dict_entrada:
                base_entrada[i][index_token] = dict_entrada[base_entrada[i][index_token]]
            else:
                base_entrada[i][index_token] = dict_entrada['<desconhecido>']
        base_entrada[i] += [0] * (50 - len(base_entrada[i]))

        pbar.update(1)
    pbar.close()

    return base_entrada, base_saida