"""
Training utilities for handwritten character recognition.
Contains training loops, model saving/loading, and evaluation functions.
"""

import os
import time
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=25, device='cpu', save_dir='model_checkpoints', 
                model_name='model', verbose=True, save_every_n_epochs=None):
    """
    Train a model with comprehensive logging and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
        model_name: Name for saving checkpoints
        verbose: Whether to print training progress
        save_every_n_epochs: Save checkpoint every N epochs (optional)
    
    Returns:
        tuple: (trained_model, training_history)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if verbose:
            print(f"Created directory: {save_dir}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking variables
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Store training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }

    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Skip if dataloader is None or empty
            if dataloader is None or len(dataloader.dataset) == 0:
                if verbose:
                    print(f"Skipping {phase} phase as dataloader is None or dataset is empty.")
                continue

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            # Calculate epoch metrics
            if total_samples == 0:
                epoch_loss = 0
                epoch_acc = 0
                if verbose:
                    print(f"No samples found for {phase} phase in epoch {epoch+1}.")
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

            if verbose:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:  # phase == 'val'
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Check if this is the best model so far
                if epoch_acc > best_acc and total_samples > 0:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    
                    # Save best model checkpoint
                    best_model_path = os.path.join(save_dir, f'best_{model_name}.pth')
                    save_checkpoint(model, optimizer, scheduler, epoch + 1, 
                                  epoch_loss, best_acc.item(), best_model_path)
                    if verbose:
                        print(f"Best model saved to {best_model_path} (Epoch {best_epoch}, Val Acc: {best_acc:.4f})")

        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1] if history['val_loss'] else history['train_loss'][-1])
            else:
                scheduler.step()
        
        # Store learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        if verbose:
            print(f'Learning Rate: {current_lr:.2e}')
            print()

        # Save periodic checkpoint
        if save_every_n_epochs is not None and (epoch + 1) % save_every_n_epochs == 0:
            periodic_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch + 1, 
                          history['val_loss'][-1] if history['val_loss'] else history['train_loss'][-1],
                          history['val_acc'][-1] if history['val_acc'] else history['train_acc'][-1],
                          periodic_path)
            if verbose:
                print(f"Periodic checkpoint saved to {periodic_path}")

    # Training complete
    time_elapsed = time.time() - start_time
    if verbose:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')

    # Load best model weights
    if best_acc > 0:
        model.load_state_dict(best_model_wts)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'final_{model_name}.pth')
    save_checkpoint(model, optimizer, scheduler, num_epochs,
                   history['val_loss'][-1] if history['val_loss'] else history['train_loss'][-1],
                   history['val_acc'][-1] if history['val_acc'] else history['train_acc'][-1],
                   final_model_path)
    if verbose:
        print(f"Final model saved to {final_model_path}")
    
    return model, history


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
        'val_acc': accuracy  # For backward compatibility
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint with comprehensive error handling.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
    
    Returns:
        tuple: (model, optimizer, scheduler, epoch, loss, accuracy)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist. Returning initial model.")
        return model, optimizer, scheduler, 0, 0.0, 0.0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer will be reinitialized.")
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}. Scheduler may be reinitialized.")
                
        # Extract metadata
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        accuracy = checkpoint.get('accuracy', checkpoint.get('val_acc', 0.0))  # Handle both keys
        
        print(f"Model loaded from {checkpoint_path}. Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return model, optimizer, scheduler, epoch, loss, accuracy
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model, optimizer, scheduler, 0, 0.0, 0.0


def test_model(model, test_loader, criterion=None, device='cpu', verbose=True):
    """
    Evaluate model on test dataset.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        criterion: Loss function (optional)
        device: Device to run evaluation on
        verbose: Whether to print results
    
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    if test_loader is None or len(test_loader.dataset) == 0:
        if verbose:
            print("Test loader is None or dataset is empty. Skipping testing.")
        return 0.0, 0.0
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No need to track gradients during testing
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    if total_samples == 0:
        if verbose:
            print("No samples found in the test set.")
        return 0.0, 0.0
        
    test_loss = running_loss / total_samples
    test_acc = running_corrects.double() / total_samples

    if verbose:
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc.item()


def plot_training_history(history, save_path=None):
    """
    Plot training history including loss, accuracy, and learning rate.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    if history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training and validation accuracy
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    if history['val_acc']:
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    if history['learning_rates']:
        axes[1, 0].plot(history['learning_rates'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot combined loss and accuracy on same plot (normalized)
    ax2 = axes[1, 1]
    if history['train_loss'] and history['train_acc']:
        # Normalize values to 0-1 range for comparison
        max_loss = max(max(history['train_loss']), max(history['val_loss']) if history['val_loss'] else 0)
        norm_train_loss = [x/max_loss for x in history['train_loss']]
        norm_val_loss = [x/max_loss for x in history['val_loss']] if history['val_loss'] else []
        
        ax2.plot(norm_train_loss, label='Normalized Train Loss', color='lightblue', linestyle='--')
        if norm_val_loss:
            ax2.plot(norm_val_loss, label='Normalized Val Loss', color='lightcoral', linestyle='--')
        ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
        if history['val_acc']:
            ax2.plot(history['val_acc'], label='Val Accuracy', color='red')
        
        ax2.set_title('Training Overview')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def get_optimizer(optimizer_name, model_parameters, learning_rate=0.001, **kwargs):
    """
    Get optimizer by name with common configurations.
    
    Args:
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw', 'rmsprop')
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate, 
                         weight_decay=kwargs.get('weight_decay', 0))
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, 
                        momentum=kwargs.get('momentum', 0.9),
                        weight_decay=kwargs.get('weight_decay', 0))
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=learning_rate,
                          weight_decay=kwargs.get('weight_decay', 0.01))
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate,
                            weight_decay=kwargs.get('weight_decay', 0))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Get learning rate scheduler by name.
    
    Args:
        scheduler_name: Name of scheduler
        optimizer: Optimizer to apply scheduler to
        **kwargs: Scheduler-specific arguments
    
    Returns:
        torch.optim.lr_scheduler: Configured scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        return lr_scheduler.StepLR(optimizer, 
                                  step_size=kwargs.get('step_size', 7),
                                  gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'multistep':
        return lr_scheduler.MultiStepLR(optimizer,
                                       milestones=kwargs.get('milestones', [10, 20]),
                                       gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer,
                                         gamma=kwargs.get('gamma', 0.95))
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer,
                                             T_max=kwargs.get('T_max', 50),
                                             eta_min=kwargs.get('eta_min', 1e-6))
    elif scheduler_name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer,
                                             mode=kwargs.get('mode', 'min'),
                                             factor=kwargs.get('factor', 0.1),
                                             patience=kwargs.get('patience', 3),
                                             verbose=kwargs.get('verbose', True))
    elif scheduler_name == 'onecycle':
        return lr_scheduler.OneCycleLR(optimizer,
                                      max_lr=kwargs.get('max_lr', 0.01),
                                      epochs=kwargs.get('epochs', 50),
                                      steps_per_epoch=kwargs.get('steps_per_epoch', 100))
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def setup_training_components(model, optimizer_config, scheduler_config=None, device='cpu'):
    """
    Setup training components (criterion, optimizer, scheduler) with configurations.
    
    Args:
        model: PyTorch model
        optimizer_config: Dictionary with optimizer configuration
        scheduler_config: Dictionary with scheduler configuration (optional)
        device: Device to use
    
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    # Setup criterion
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    if isinstance(optimizer_config, dict):
        optimizer_name = optimizer_config.pop('name', 'adam')
        optimizer = get_optimizer(optimizer_name, model.parameters(), **optimizer_config)
    else:
        optimizer = optimizer_config  # Assume it's already an optimizer instance
    
    # Setup scheduler
    scheduler = None
    if scheduler_config:
        if isinstance(scheduler_config, dict):
            scheduler_name = scheduler_config.pop('name')
            scheduler = get_scheduler(scheduler_name, optimizer, **scheduler_config)
        else:
            scheduler = scheduler_config  # Assume it's already a scheduler instance
    
    return criterion, optimizer, scheduler


def evaluate_model_detailed(model, test_loader, class_names, device='cpu'):
    """
    Detailed evaluation with per-class metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use
    
    Returns:
        dict: Detailed evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    num_classes = len(class_names)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[class_names[i]] = class_correct[i] / class_total[i]
        else:
            class_accuracy[class_names[i]] = 0.0
    
    # Overall accuracy
    overall_accuracy = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracy': class_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'class_totals': dict(zip(class_names, class_total))
    }
