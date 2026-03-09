"""
AgroSmart - Script de Treinamento do Modelo CNN
=================================================
Este script treina uma Rede Neural Convolucional para classificar
imagens de folhas como "saudável" ou "doente", utilizando Transfer
Learning com MobileNetV2 e aceleração por GPU (NVIDIA RTX 2070).

Uso:
    python src/treinar_modelo.py
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# Configurações gerais do projeto
# ============================================================
DIRETORIO_DATASET = os.path.join(os.path.dirname(__file__), '..', 'dataset')
DIRETORIO_MODELOS = os.path.join(os.path.dirname(__file__), '..', 'models')
CAMINHO_MODELO = os.path.join(DIRETORIO_MODELOS, 'agrosmart_model.h5')

TAMANHO_IMAGEM = (224, 224)   # Tamanho esperado pelo MobileNetV2
BATCH_SIZE = 32
EPOCAS = 20
TAXA_APRENDIZADO = 0.0001


def configurar_gpu():
    """
    Configura o TensorFlow para utilizar a GPU NVIDIA disponível,
    com crescimento dinâmico de memória para evitar alocação excessiva.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")
            print("[INFO] Crescimento dinâmico de memória ativado.")
        except RuntimeError as e:
            print(f"[ERRO] Falha ao configurar GPU: {e}")
    else:
        print("[AVISO] Nenhuma GPU encontrada. O treinamento será feito na CPU.")


def validar_dataset(diretorio):
    """
    Verifica se o diretório do dataset existe e contém as subpastas
    esperadas ('saudavel' e 'doente') com pelo menos uma imagem cada.
    """
    if not os.path.isdir(diretorio):
        print(f"[ERRO] Diretório do dataset não encontrado: {diretorio}")
        sys.exit(1)

    subpastas = os.listdir(diretorio)
    for classe in ['saudavel', 'doente']:
        caminho_classe = os.path.join(diretorio, classe)
        if not os.path.isdir(caminho_classe):
            print(f"[ERRO] Subpasta '{classe}' não encontrada em {diretorio}")
            sys.exit(1)
        arquivos = [f for f in os.listdir(caminho_classe)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(arquivos) == 0:
            print(f"[ERRO] Nenhuma imagem encontrada em {caminho_classe}")
            sys.exit(1)
        print(f"[INFO] Classe '{classe}': {len(arquivos)} imagem(ns) encontrada(s).")


def criar_geradores(diretorio):
    """
    Cria geradores de dados para treino e validação com aumento de dados
    (data augmentation) para melhorar a generalização do modelo.
    """
    # Gerador de treino com aumento de dados
    gerador_treino = ImageDataGenerator(
        rescale=1.0 / 255.0,           # Normalização [0, 1]
        rotation_range=30,              # Rotação aleatória
        width_shift_range=0.2,          # Deslocamento horizontal
        height_shift_range=0.2,         # Deslocamento vertical
        shear_range=0.2,                # Cisalhamento
        zoom_range=0.2,                 # Zoom aleatório
        horizontal_flip=True,           # Espelhamento horizontal
        fill_mode='nearest',
        validation_split=0.2            # 20% para validação
    )

    dados_treino = gerador_treino.flow_from_directory(
        diretorio,
        target_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        class_mode='binary',            # Classificação binária
        subset='training',
        shuffle=True
    )

    dados_validacao = gerador_treino.flow_from_directory(
        diretorio,
        target_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return dados_treino, dados_validacao


def construir_modelo():
    """
    Constrói o modelo utilizando Transfer Learning com MobileNetV2.
    As camadas da base são congeladas para aproveitar os pesos
    pré-treinados do ImageNet.
    """
    # Carrega a base MobileNetV2 sem a camada de classificação
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(TAMANHO_IMAGEM[0], TAMANHO_IMAGEM[1], 3)
    )

    # Congela as camadas da base para manter os pesos pré-treinados
    base.trainable = False

    # Adiciona camadas de classificação personalizadas
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)                       # Regularização
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    saida = Dense(1, activation='sigmoid')(x)  # Saída binária

    modelo = Model(inputs=base.input, outputs=saida)

    # Compila o modelo com otimizador Adam
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    modelo.summary()
    return modelo


def treinar(modelo, dados_treino, dados_validacao):
    """
    Executa o treinamento do modelo com callbacks para parada antecipada
    e salvamento do melhor modelo durante o treinamento.
    """
    # Garante que o diretório de modelos exista
    os.makedirs(DIRETORIO_MODELOS, exist_ok=True)

    callbacks = [
        # Para o treinamento se a val_loss não melhorar por 5 épocas
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Salva o melhor modelo baseado na val_accuracy
        ModelCheckpoint(
            CAMINHO_MODELO,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    historico = modelo.fit(
        dados_treino,
        validation_data=dados_validacao,
        epochs=EPOCAS,
        callbacks=callbacks,
        verbose=1
    )

    return historico


def main():
    """Função principal que orquestra todo o pipeline de treinamento."""
    print("=" * 60)
    print("  AgroSmart - Treinamento do Modelo de Classificação")
    print("=" * 60)

    # 1. Configura GPU
    configurar_gpu()

    # 2. Valida o dataset
    diretorio_abs = os.path.abspath(DIRETORIO_DATASET)
    print(f"\n[INFO] Dataset: {diretorio_abs}")
    validar_dataset(diretorio_abs)

    # 3. Cria geradores de dados
    print("\n[INFO] Preparando geradores de dados...")
    dados_treino, dados_validacao = criar_geradores(diretorio_abs)

    # Exibe o mapeamento de classes
    print(f"[INFO] Mapeamento de classes: {dados_treino.class_indices}")

    # 4. Constrói o modelo
    print("\n[INFO] Construindo modelo com Transfer Learning (MobileNetV2)...")
    modelo = construir_modelo()

    # 5. Treina o modelo
    print("\n[INFO] Iniciando treinamento...")
    historico = treinar(modelo, dados_treino, dados_validacao)

    # 6. Salva o modelo final
    modelo.save(CAMINHO_MODELO)
    print(f"\n[INFO] Modelo salvo com sucesso em: {os.path.abspath(CAMINHO_MODELO)}")

    # 7. Exibe métricas finais
    val_acc = historico.history.get('val_accuracy', [0])[-1]
    val_loss = historico.history.get('val_loss', [0])[-1]
    print(f"[INFO] Acurácia de validação final: {val_acc:.4f}")
    print(f"[INFO] Loss de validação final   : {val_loss:.4f}")
    print("\n[INFO] Treinamento concluído com sucesso!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO FATAL] Ocorreu um erro inesperado: {e}")
        sys.exit(1)
