#!/usr/bin/env python3
"""
Quick Start Script for Ultimate Deep Learning System
Run this to train the 2000+ layer model
"""

import os
import sys

print("="*80)
print("ULTIMATE DEEP LEARNING SYSTEM - QUICK START")
print("="*80)
print("\nThis will train a 2000+ layer deep neural network with:")
print("  - ResNet blocks (residual connections)")
print("  - DenseNet blocks (dense connections)")
print("  - Attention mechanisms")
print("  - 200+ engineered features")
print("\n⚠️  WARNING: This will take significant time and GPU resources")
print("   Estimated time: 16-32 hours")
print("   GPU Memory needed: 8-16 GB")
print("\n" + "="*80)

response = input("\nContinue? (yes/no): ")

if response.lower() not in ['yes', 'y']:
    print("Cancelled.")
    sys.exit(0)

print("\n" + "="*80)
print("Starting training...")
print("="*80)
print("\n💡 Tips:")
print("  - Monitor progress in terminal")
print("  - Check 'ultimate_deep_advanced_model.h5' for saved model")
print("  - Use Ctrl+C to stop (model will be saved if checkpointed)")
print("\n" + "="*80 + "\n")

# Import and run
try:
    import ultimate_deep_learning_advanced
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("   Check for saved model: ultimate_deep_advanced_model.h5")
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

