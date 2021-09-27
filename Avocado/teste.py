import pandas as pd
import pickle

import matplotlib.pyplot as plt
try:
    arquivo_teste = pd.DataFrame(pd.read_csv("teste.csv", sep=","))

except FileNotFoundError:
    print("ARQUIVO NAO FOI ENCONTRADO")

except Exception as e:
    print(f"ERRO AO SUBIR ARQUIVO: {e}")

modelo_carregado = pickle.load(open("modelo_finalizado.sav", "rb"))

# REALIZANDO O PREDICT DO ARQUIVO DE TESTE COM O MODELO IMPORTADO

X = arquivo_teste[["TOTAL VOLUME", "4046", "4225", "4770", "TOTAL BAGS", "TYPE", "REGION"]]

arquivo_teste["RESULTADO"] = modelo_carregado.predict(X)

print(arquivo_teste)
