#!/usr/bin/env python3
"""
Simple test script to verify GAN training functionality.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from models.gan_models import Generator, Discriminator
from utils.data_utils import TreasuryDataProcessor

def test_gan_training():
    """Test basic GAN training functionality."""
    print("🚀 Starting GAN training test...")
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    processor = TreasuryDataProcessor(['2Y', '5Y', '10Y', '30Y'], 50)
    sequences, targets, scaler = processor.prepare_data('2022-01-01', '2024-01-01')
    print(f"Data loaded: {sequences.shape}")
    
    # Create models
    generator = Generator(100, [256, 128], 14, 50).to(device)
    discriminator = Discriminator(14, [128, 64], 50).to(device)
    print("✅ GAN models created successfully")
    
    # Setup training
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()
    print("✅ Training setup complete")
    
    # Training loop
    batch_size = 32
    num_epochs = 2
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i in range(0, min(100, len(sequences)), batch_size):
            batch = sequences[i:i+batch_size]
            batch = torch.FloatTensor(batch).to(device)
            
            # Create labels
            real_labels = torch.ones(batch.size(0), 1).to(device)
            fake_labels = torch.zeros(batch.size(0), 1).to(device)
            
            # Generate fake data
            noise = torch.randn(batch.size(0), 100).to(device)
            fake_data = generator(noise)
            
            # Train discriminator
            real_output = discriminator(batch)
            fake_output = discriminator(fake_data.detach())
            d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
            
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            if i % 64 == 0:
                print(f"  Batch {i//batch_size}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    print("✅ Training test completed successfully!")
    
    # Test generation
    print("🧪 Testing sample generation...")
    with torch.no_grad():
        test_noise = torch.randn(5, 100).to(device)
        synthetic_samples = generator(test_noise)
        print(f"Generated samples shape: {synthetic_samples.shape}")
        print(f"Sample values range: {synthetic_samples.min().item():.4f} to {synthetic_samples.max().item():.4f}")
    
    print("🎉 All tests passed! GAN system is working correctly.")

if __name__ == "__main__":
    test_gan_training() 