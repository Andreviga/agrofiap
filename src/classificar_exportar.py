"""
AgroSmart - Script de Classificação e Exportação de Relatório
==============================================================
Este script carrega o modelo treinado, classifica novas imagens
de folhas como "saudável" ou "doente" e exporta um relatório
em formato CSV com os resultados.

Uso:
    python src/classificar_exportar.py                           (classifica todas as imagens em dataset/)
    python src/classificar_exportar.py imagem1.jpg imagem2.png   (classifica imagens específicas)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ============================================================
# Configurações gerais do projeto
# ============================================================
DIRETORIO_BASE = os.path.join(os.path.dirname(__file__), '..')
CAMINHO_MODELO = os.path.join(DIRETORIO_BASE, 'models', 'agrosmart_model.h5')
DIRETORIO_DATASET = os.path.join(DIRETORIO_BASE, 'dataset')
CAMINHO_RELATORIO = os.path.join(DIRETORIO_BASE, 'relatorio_classificacao.csv')

TAMANHO_IMAGEM = (224, 224)  # Deve ser o mesmo usado no treinamento

# Mapeamento de classes (deve corresponder ao treinamento)
# O ImageDataGenerator ordena as pastas alfabeticamente: doente=0, saudavel=1
CLASSES = {0: 'doente', 1: 'saudavel'}


def carregar_modelo(caminho):
    """
    Carrega o modelo Keras salvo em disco.
    Trata erros caso o arquivo não exista ou esteja corrompido.
    """
    if not os.path.isfile(caminho):
        print(f"[ERRO] Modelo não encontrado em: {os.path.abspath(caminho)}")
        print("[DICA] Execute primeiro o script de treinamento: python src/treinar_modelo.py")
        sys.exit(1)

    try:
        modelo = load_model(caminho)
        print(f"[INFO] Modelo carregado com sucesso: {os.path.abspath(caminho)}")
        return modelo
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o modelo: {e}")
        sys.exit(1)


def preprocessar_imagem(caminho_imagem):
    """
    Lê uma imagem com OpenCV, converte de BGR para RGB,
    redimensiona e normaliza para o formato esperado pelo modelo.

    Retorna:
        numpy array com shape (1, 224, 224, 3) normalizado entre [0, 1]
        ou None se houver erro na leitura.
    """
    try:
        # OpenCV lê imagens em formato BGR
        imagem_bgr = cv2.imread(caminho_imagem)

        if imagem_bgr is None:
            print(f"[AVISO] Não foi possível ler a imagem: {caminho_imagem}")
            return None

        # Converte BGR (OpenCV) para RGB (TensorFlow/Keras)
        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

        # Redimensiona para o tamanho esperado pelo modelo
        imagem_redimensionada = cv2.resize(imagem_rgb, TAMANHO_IMAGEM)

        # Normaliza os pixels para o intervalo [0, 1]
        imagem_normalizada = imagem_redimensionada.astype('float32') / 255.0

        # Adiciona a dimensão do batch: (224, 224, 3) -> (1, 224, 224, 3)
        imagem_batch = np.expand_dims(imagem_normalizada, axis=0)

        return imagem_batch

    except Exception as e:
        print(f"[AVISO] Erro ao processar imagem '{caminho_imagem}': {e}")
        return None


def classificar_imagem(modelo, imagem_processada):
    """
    Realiza a predição sobre uma imagem pré-processada.

    Retorna:
        (categoria, confianca): tupla com a classe predita e o nível
        de confiança (acurácia) da predição.
    """
    # O modelo retorna um valor entre 0 e 1 (sigmoid)
    predicao = modelo.predict(imagem_processada, verbose=0)
    valor = float(predicao[0][0])

    # Se valor >= 0.5, classe 1 (saudavel); caso contrário, classe 0 (doente)
    if valor >= 0.5:
        categoria = CLASSES[1]   # saudavel
        confianca = valor        # Confiança na classe "saudável"
    else:
        categoria = CLASSES[0]   # doente
        confianca = 1.0 - valor  # Confiança na classe "doente"

    return categoria, round(confianca, 4)


def coletar_imagens_do_dataset(diretorio):
    """
    Percorre todas as subpastas do dataset e coleta os caminhos
    das imagens encontradas.
    """
    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp')
    imagens = []

    if not os.path.isdir(diretorio):
        print(f"[AVISO] Diretório do dataset não encontrado: {diretorio}")
        return imagens

    for raiz, _, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.lower().endswith(extensoes_validas):
                imagens.append(os.path.join(raiz, arquivo))

    return sorted(imagens)


def exportar_relatorio(resultados, caminho_csv):
    """
    Exporta os resultados da classificação em um arquivo CSV
    com as colunas: nome_da_imagem, categoria_detectada, acuracia.
    """
    try:
        df = pd.DataFrame(resultados, columns=[
            'nome_da_imagem',
            'categoria_detectada',
            'acuracia'
        ])

        df.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] Relatório CSV exportado com sucesso: {os.path.abspath(caminho_csv)}")
        print(f"[INFO] Total de imagens classificadas: {len(resultados)}")

        # Exibe um resumo no terminal
        print("\n" + "=" * 60)
        print("  Resumo da Classificação")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)

    except Exception as e:
        print(f"[ERRO] Falha ao exportar relatório CSV: {e}")
        sys.exit(1)


def main():
    """Função principal que orquestra o pipeline de classificação e exportação."""
    print("=" * 60)
    print("  AgroSmart - Classificação e Exportação de Relatório")
    print("=" * 60)

    # 1. Carrega o modelo treinado
    modelo = carregar_modelo(CAMINHO_MODELO)

    # 2. Coleta a lista de imagens para classificação
    if len(sys.argv) > 1:
        # Imagens passadas como argumentos na linha de comando
        lista_imagens = sys.argv[1:]
        print(f"\n[INFO] {len(lista_imagens)} imagem(ns) recebida(s) via argumentos.")
    else:
        # Busca todas as imagens do diretório dataset
        lista_imagens = coletar_imagens_do_dataset(
            os.path.abspath(DIRETORIO_DATASET)
        )
        print(f"\n[INFO] {len(lista_imagens)} imagem(ns) encontrada(s) no dataset.")

    if not lista_imagens:
        print("[AVISO] Nenhuma imagem encontrada para classificação.")
        sys.exit(0)

    # 3. Classifica cada imagem e armazena os resultados
    resultados = []

    for caminho in lista_imagens:
        nome_arquivo = os.path.basename(caminho)

        # Verifica se o arquivo existe
        if not os.path.isfile(caminho):
            print(f"[AVISO] Arquivo não encontrado, pulando: {caminho}")
            continue

        # Pré-processa a imagem
        imagem_processada = preprocessar_imagem(caminho)
        if imagem_processada is None:
            continue

        # Classifica
        try:
            categoria, confianca = classificar_imagem(modelo, imagem_processada)
            resultados.append([nome_arquivo, categoria, confianca])
            print(f"  -> {nome_arquivo}: {categoria} (confiança: {confianca})")
        except Exception as e:
            print(f"[AVISO] Erro ao classificar '{nome_arquivo}': {e}")

    # 4. Exporta o relatório CSV (requisito crítico)
    if resultados:
        exportar_relatorio(resultados, os.path.abspath(CAMINHO_RELATORIO))
    else:
        print("[AVISO] Nenhuma imagem foi classificada com sucesso. Relatório não gerado.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO FATAL] Ocorreu um erro inesperado: {e}")
        sys.exit(1)
