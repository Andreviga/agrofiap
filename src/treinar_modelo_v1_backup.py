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
    Cria geradores de dados para treino e validação com data augmentation
    robusto, incluindo ajuste de brilho (essencial para fotos ao ar livre
    no campo), rotação, zoom e espelhamento.
    """
    # Gerador de TREINO com augmentation agressivo para generalização
    gerador_treino = ImageDataGenerator(
        rescale=1.0 / 255.0,           # Normalização [0, 1]
        rotation_range=40,              # Rotação aleatória ampla
        width_shift_range=0.2,          # Deslocamento horizontal
        height_shift_range=0.2,         # Deslocamento vertical
        shear_range=0.2,                # Cisalhamento
        zoom_range=0.3,                 # Zoom aleatório (até 30%)
        brightness_range=[0.7, 1.3],    # Variação de brilho (fotos de campo)
        horizontal_flip=True,           # Espelhamento horizontal
        vertical_flip=True,             # Espelhamento vertical
        channel_shift_range=25.0,       # Variação sutil de cor
        fill_mode='nearest',
        validation_split=0.2            # 20% para validação
    )

    # Gerador de VALIDAÇÃO — apenas normalização, sem augmentation
    gerador_validacao = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    dados_treino = gerador_treino.flow_from_directory(
        diretorio,
        target_size=TAMANHO_IMAGEM,
        batch_size=BATCH_SIZE,
        class_mode='binary',            # Classificação binária
        subset='training',
        shuffle=True
    )

    dados_validacao = gerador_validacao.flow_from_directory(
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
    Fase 1: base inteiramente congelada, treina apenas camadas densas.
    O retorno inclui a referência à base para o fine-tuning posterior.
    """
    # Carrega a base MobileNetV2 sem a camada de classificação
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(TAMANHO_IMAGEM[0], TAMANHO_IMAGEM[1], 3)
    )

    # Congela TODAS as camadas da base (Fase 1)
    base.trainable = False

    # Camadas de classificação personalizadas (cabeça mais robusta)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)                # Estabiliza a distribuição
    x = Dropout(0.4)(x)                       # Regularização forte
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    saida = Dense(1, activation='sigmoid')(x)  # Saída binária

    modelo = Model(inputs=base.input, outputs=saida)

    # Compila para Fase 1 com learning rate padrão
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO_FASE1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"[INFO] Total de camadas do modelo: {len(modelo.layers)}")
    print(f"[INFO] Camadas da base MobileNetV2: {len(base.layers)}")
    modelo.summary()
    return modelo, base


def criar_callbacks(fase):
    """
    Cria callbacks configurados para cada fase do treinamento.
    - ModelCheckpoint: salva SOMENTE o .h5 com a melhor val_accuracy.
    - EarlyStopping: interrompe se val_loss parar de cair por 5 épocas.
    - ReduceLROnPlateau: reduz learning rate ao estagnar (decaimento).
    """
    return [
        ModelCheckpoint(
            CAMINHO_MODELO,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,              # Reduz LR pela metade
            patience=3,              # Após 3 épocas sem melhora
            min_lr=1e-7,
            verbose=1
        )
    ]


def treinar_fase1(modelo, dados_treino, dados_validacao):
    """
    FASE 1 — Treinamento das camadas densas com base congelada.
    Objetivo: ajustar a cabeça de classificação rapidamente.
    """
    os.makedirs(DIRETORIO_MODELOS, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FASE 1: Treinamento das camadas densas (base congelada)")
    print("=" * 60)

    historico = modelo.fit(
        dados_treino,
        validation_data=dados_validacao,
        epochs=EPOCAS_FASE1,
        callbacks=criar_callbacks(fase=1),
        verbose=1
    )

    val_acc = historico.history.get('val_accuracy', [0])[-1]
    print(f"\n[FASE 1] Melhor val_accuracy: {val_acc:.4f}")
    return historico


def treinar_fase2(modelo, base, dados_treino, dados_validacao):
    """
    FASE 2 — Fine-Tuning: descongela as últimas 30 camadas da MobileNetV2
    e re-treina com learning rate muito baixa (1e-5) para ajustar os pesos
    finos às imagens de folhas sem destruir os pesos pré-treinados.
    """
    print("\n" + "=" * 60)
    print(f"  FASE 2: Fine-Tuning (descongelando últimas {CAMADAS_DESCONGELAR} camadas)")
    print("=" * 60)

    # Descongela as últimas N camadas da base MobileNetV2
    base.trainable = True
    total_camadas = len(base.layers)
    for camada in base.layers[:total_camadas - CAMADAS_DESCONGELAR]:
        camada.trainable = False

    # Conta camadas treináveis vs congeladas
    treinaveis = sum(1 for c in modelo.layers if c.trainable)
    congeladas = sum(1 for c in modelo.layers if not c.trainable)
    print(f"[INFO] Camadas treináveis: {treinaveis} | Congeladas: {congeladas}")

    # Recompila com learning rate muito baixa para fine-tuning
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO_FASE2),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    historico = modelo.fit(
        dados_treino,
        validation_data=dados_validacao,
        epochs=EPOCAS_FASE2,
        callbacks=criar_callbacks(fase=2),
        verbose=1
    )

    val_acc = historico.history.get('val_accuracy', [0])[-1]
    print(f"\n[FASE 2] Melhor val_accuracy: {val_acc:.4f}")
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
    modelo, base = construir_modelo()

    # 5. FASE 1 — Treina apenas as camadas densas (base congelada)
    historico_f1 = treinar_fase1(modelo, dados_treino, dados_validacao)

    # 6. FASE 2 — Fine-tuning das últimas 30 camadas da MobileNetV2
    historico_f2 = treinar_fase2(modelo, base, dados_treino, dados_validacao)

    # 7. Salva o modelo final (o melhor já foi salvo pelo ModelCheckpoint)
    modelo.save(CAMINHO_MODELO)
    print(f"\n[INFO] Modelo final salvo em: {os.path.abspath(CAMINHO_MODELO)}")

    # 8. Exibe métricas finais de ambas as fases
    f1_acc = historico_f1.history.get('val_accuracy', [0])[-1]
    f2_acc = historico_f2.history.get('val_accuracy', [0])[-1]
    f2_loss = historico_f2.history.get('val_loss', [0])[-1]
    print(f"\n{'=' * 60}")
    print(f"  RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"  Fase 1 (camadas densas)  — val_accuracy: {f1_acc:.4f}")
    print(f"  Fase 2 (fine-tuning)     — val_accuracy: {f2_acc:.4f}")
    print(f"  Fase 2 (fine-tuning)     — val_loss:     {f2_loss:.4f}")
    print(f"{'=' * 60}")
    print("[INFO] Treinamento concluído com sucesso!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Treinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO FATAL] Ocorreu um erro inesperado: {e}")
        sys.exit(1)
