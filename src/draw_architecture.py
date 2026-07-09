import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Input
    ax.add_patch(patches.Rectangle((0.5, 2.5), 1, 1, fill=True, color='lightblue', label='Input Image (1x224x224)'))
    ax.text(1.0, 3.0, 'Input Image\n(CT Scan)', ha='center', va='center')
    
    # Autoencoder
    ax.add_patch(patches.Rectangle((2.5, 4), 2, 1, fill=True, color='lightgreen'))
    ax.text(3.5, 4.5, 'Encoder\n(Conv Layers)', ha='center', va='center')
    
    ax.add_patch(patches.Rectangle((5.5, 4), 1, 1, fill=True, color='orange'))
    ax.text(6.0, 4.5, 'Latent Space\n(Features)', ha='center', va='center')
    
    ax.add_patch(patches.Rectangle((7.5, 4), 2, 1, fill=True, color='lightgreen'))
    ax.text(8.5, 4.5, 'Decoder\n(ConvTranspose)', ha='center', va='center')
    
    ax.add_patch(patches.Rectangle((10.5, 4), 1, 1, fill=True, color='lightblue'))
    ax.text(11.0, 4.5, 'Reconstructed\nImage', ha='center', va='center')
    
    # ResNet
    ax.add_patch(patches.Rectangle((2.5, 1), 3, 1, fill=True, color='salmon'))
    ax.text(4.0, 1.5, 'ResNet18\n(Feature Extractor)', ha='center', va='center')
    
    # Concat
    ax.add_patch(patches.Rectangle((6.5, 2.5), 1, 1, fill=True, color='yellow'))
    ax.text(7.0, 3.0, 'Concatenate', ha='center', va='center')
    
    # Classifier
    ax.add_patch(patches.Rectangle((8.5, 2.5), 2, 1, fill=True, color='purple', alpha=0.5))
    ax.text(9.5, 3.0, 'Fully Connected\nClassifier', ha='center', va='center')
    
    # Output
    ax.add_patch(patches.Rectangle((11.5, 2.5), 1, 1, fill=True, color='pink'))
    ax.text(12.0, 3.0, 'Output\n(COVID / Normal)', ha='center', va='center')
    
    # Arrows
    ax.annotate('', xy=(2.5, 4.5), xytext=(1.5, 3.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(2.5, 1.5), xytext=(1.5, 3.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(5.5, 4.5), xytext=(4.5, 4.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(7.5, 4.5), xytext=(6.5, 4.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(10.5, 4.5), xytext=(9.5, 4.5), arrowprops=dict(arrowstyle="->"))
    
    # From latent and resnet to concat
    ax.annotate('', xy=(6.5, 3.0), xytext=(6.0, 4.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(6.5, 2.8), xytext=(5.5, 1.5), arrowprops=dict(arrowstyle="->"))
    
    ax.annotate('', xy=(8.5, 3.0), xytext=(7.5, 3.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(11.5, 3.0), xytext=(10.5, 3.0), arrowprops=dict(arrowstyle="->"))
    
    plt.xlim(0, 13)
    plt.ylim(0, 6)
    plt.axis('off')
    plt.title("Hybrid Autoencoder + ResNet18 Architecture")
    
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    plt.savefig(os.path.join(docs_dir, "architecture.png"), bbox_inches='tight')
    print("Architecture diagram saved to docs/architecture.png")

if __name__ == "__main__":
    draw_architecture()
