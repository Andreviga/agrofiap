# agrofiap

**AgroSmart** - Sistema de Visão Computacional para classificação de folhas como **saudável** ou **doente**, utilizando Transfer Learning com MobileNetV2.

## Estrutura do Projeto

```
agrofiap/
├── dataset/
│   ├── saudavel/       ← imagens de folhas saudáveis
│   └── doente/         ← imagens de folhas doentes
├── models/             ← modelo treinado (.h5)
├── src/
│   ├── treinar_modelo.py
│   └── classificar_exportar.py
└── requirements.txt
```

## Como usar

```bash
pip install -r requirements.txt
python src/treinar_modelo.py
python src/classificar_exportar.py
```
