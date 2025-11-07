import time
import os
import subprocess

print("="*80)
print("MONITORING DEEP LEARNING TRAINING")
print("="*80)

log_file = "deep_learning_training.log"
max_wait = 3600  # 1 hour max

print(f"\nMonitoring: {log_file}")
print("Press Ctrl+C to stop monitoring\n")

start_time = time.time()
last_size = 0

try:
    while True:
        if os.path.exists(log_file):
            # Check if file is growing
            current_size = os.path.getsize(log_file)
            
            if current_size > last_size:
                # File is being written to
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Show last 10 lines
                    print("\n" + "="*80)
                    print(f"Last update (file size: {current_size} bytes):")
                    print("="*80)
                    for line in lines[-10:]:
                        if any(keyword in line for keyword in ['Epoch', 'accuracy', 'loss', 'features', 'Training', 'COMPLETE', 'ERROR']):
                            print(line.strip())
                last_size = current_size
            else:
                print(".", end="", flush=True)
        else:
            print("Waiting for log file to be created...")
        
        # Check if training completed
        if os.path.exists("best_deep_model.h5"):
            print("\n\n✅ Training completed! Model saved: best_deep_model.h5")
            break
        
        # Check elapsed time
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"\n\n⏱️  Max wait time reached ({max_wait}s)")
            break
        
        time.sleep(10)  # Check every 10 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user")

print("\n" + "="*80)
print("Monitoring complete")
print("="*80)

