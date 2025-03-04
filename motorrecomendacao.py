import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#Carregar e ler o atrquivo csv
dados = pd.read_csv('motor_recomendacao-1.csv')
print(dados.columns)

#Criando uma tabela dinamica
test = dados.pivot_table(index='usuario_id', columns='item_id', values='avaliacao')
print(test)

#Substituir os valores alzentes das avaliações pela media
#test['avaliacao'].fillna(media, inplace=True)
test.fillna(test.mean().mean(), inplace=True)
print(test)

#Comparar usuarios com base nas avaliações de cada item 
print(test)

#Usuario 1 
usuarioespecifico1 = 1
usuario1 = test.loc[usuarioespecifico1].values.reshape(1,-1)
print(usuario1)

#Usuario 2
usuarioespecifico2 = 2
usuario2 = test.loc[usuarioespecifico2].values.reshape(1,-1)
print(usuario2)

#usuario 3
usuarioespecifico3 = 3
usuario3 = test.loc[usuarioespecifico3].values.reshape(1,-1)
print(usuario3)

#usuario 4
usuarioespecifico4 = 4
usuario4 = test.loc[usuarioespecifico4].values.reshape(1,-1)
print(usuario4)

#usuario 5
usuarioespecifico5 = 5
usuario5 = test.loc[usuarioespecifico5].values.reshape(1,-1)
print(usuario5)

#Usuário 1 ↔ Usuário 2
similaridade1_2 = cosine_similarity(usuario1, usuario2)
print('similaridade entre usuario 1 e 2')
print(similaridade1_2)

#Usuário 1 ↔ Usuário 3
similaridade1_3 = cosine_similarity(usuario1, usuario3)
print('similaridade entre usuario 1 e 3')
print(similaridade1_3)

#Usuário 1 ↔ Usuário 4
similaridade1_4 = cosine_similarity(usuario1, usuario4)
print('similaridade entre usuario 1 e 4')
print(similaridade1_4)

#Usuário 1 ↔ Usuário 5
similaridade1_5 = cosine_similarity(usuario1, usuario5)
print('similaridade entre usuario 1 e 5')
print(similaridade1_5)
#Usuário 2 ↔ Usuário 3
similaridade2_3 = cosine_similarity(usuario2, usuario3)
print('similaridade entre usuario 2 e 3')
print(similaridade2_3)
#Usuário 2 ↔ Usuário 4
similaridade2_4 = cosine_similarity(usuario2, usuario4)
print('similaridade entre usuario 2 e 4')
print(similaridade2_4)
#Usuário 2 ↔ Usuário 5
similaridade2_5 = cosine_similarity(usuario2, usuario5)
print('similaridade entre usuario 2 e 5')
print(similaridade2_5)
#Usuário 3 ↔ Usuário 4
similaridade3_4 = cosine_similarity(usuario3, usuario4)
print('similaridade entre usuario 3 e 4')
print(similaridade3_4)
#Usuário 3 ↔ Usuário 5
similaridade3_5 = cosine_similarity(usuario3, usuario5)
print('similaridade entre usuario 3 e 5')
print(similaridade3_5)
#Usuário 4 ↔ Usuário 5
similaridade4_5 = cosine_similarity(usuario4, usuario5)
print('similaridade entre usuario 4 e 5')
print(similaridade4_5)
