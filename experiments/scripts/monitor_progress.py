"""
Monitor the progress of the evaluation script in real-time.
"""

import time
import os
from pathlib import Path

def get_process_status():
    """Check if the process is still running"""
    result = os.popen('ps aux | grep "run_stages_2_3_BETTER_MODEL" | grep -v grep | grep -v monitor').read()
    return bool(result.strip())

def parse_log_file(log_path):
    """Parse the log file and extract key information"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    info = {
        'model_loaded': 'Model loaded successfully!' in content or '✅ MLX model initialized' in content,
        'stage2_started': 'STAGE 2:' in content,
        'stage2_complete': 'STAGE 2 COMPLETE' in content,
        'stage3_started': 'STAGE 3:' in content,
        'stage3_complete': 'STAGE 3 COMPLETE' in content,
        'evaluation_complete': 'EVALUATION COMPLETE' in content,
        'error': 'Error' in content or 'Traceback' in content,
        'total_lines': len(content.split('\n'))
    }
    
    # Count samples processed
    info['stage2_samples'] = content.count('Sample ') if info['stage2_started'] else 0
    info['stage3_samples'] = content.count('Sample ') - info['stage2_samples'] if info['stage3_started'] else 0
    
    # Get last few lines
    lines = content.split('\n')
    info['last_lines'] = '\n'.join([l for l in lines[-10:] if l.strip()])
    
    return info

def print_status_bar(current, total, label="Progress"):
    """Print a nice progress bar"""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100 * current / total
    print(f"  {label}: [{bar}] {percent:.0f}% ({current}/{total})")

def main():
    log_path = Path("better_model_output.log")
    
    print("\n" + "="*80)
    print("  📊 MONITORING EVALUATION PROGRESS")
    print("="*80 + "\n")
    
    start_time = time.time()
    last_line_count = 0
    
    try:
        while True:
            # Check if process is still running
            is_running = get_process_status()
            elapsed = time.time() - start_time
            
            # Parse log file
            info = parse_log_file(log_path)
            
            # Clear screen for clean updates (optional)
            # os.system('clear')
            
            print(f"\n{'─'*80}")
            print(f"⏱️  Elapsed Time: {int(elapsed//60)}m {int(elapsed%60)}s")
            print(f"🔄 Process Status: {'✅ Running' if is_running else '⚠️  Stopped'}")
            print(f"{'─'*80}\n")
            
            if info:
                # Model loading status
                if info['model_loaded']:
                    print("✅ Model Loaded: Mistral-7B-Instruct-v0.2")
                else:
                    print("📥 Model Loading: In progress...")
                
                print()
                
                # Stage 2 status
                if info['stage2_complete']:
                    print("✅ Stage 2 (M4 Captioning): COMPLETE")
                    print_status_bar(3, 3, "Stage 2 Samples")
                elif info['stage2_started']:
                    samples = min(info['stage2_samples'], 3)
                    print(f"🔄 Stage 2 (M4 Captioning): Sample {samples}/3")
                    print_status_bar(samples, 3, "Stage 2 Samples")
                else:
                    print("⏳ Stage 2 (M4 Captioning): Waiting to start...")
                
                print()
                
                # Stage 3 status
                if info['stage3_complete']:
                    print("✅ Stage 3 (HAR CoT): COMPLETE")
                    print_status_bar(3, 3, "Stage 3 Samples")
                elif info['stage3_started']:
                    samples = min(info['stage3_samples'], 3)
                    print(f"🔄 Stage 3 (HAR CoT): Sample {samples}/3")
                    print_status_bar(samples, 3, "Stage 3 Samples")
                else:
                    print("⏳ Stage 3 (HAR CoT): Waiting to start...")
                
                print()
                
                # Overall progress
                total_steps = 6  # 3 stage2 + 3 stage3
                completed_steps = 0
                if info['stage2_complete']:
                    completed_steps += 3
                elif info['stage2_started']:
                    completed_steps += min(info['stage2_samples'], 3)
                
                if info['stage3_complete']:
                    completed_steps += 3
                elif info['stage3_started']:
                    completed_steps += min(info['stage3_samples'], 3)
                
                print_status_bar(completed_steps, total_steps, "Overall Progress")
                
                print()
                print(f"{'─'*80}")
                print("📄 Recent Log Output:")
                print(f"{'─'*80}")
                
                # Show last few lines
                recent = info['last_lines']
                if recent:
                    # Truncate very long lines
                    lines = recent.split('\n')
                    for line in lines[-5:]:
                        if line.strip():
                            truncated = line[:76] + "..." if len(line) > 76 else line
                            print(f"  {truncated}")
                
                # Check if complete
                if info['evaluation_complete']:
                    print("\n" + "="*80)
                    print("  🎉 EVALUATION COMPLETE!")
                    print("="*80 + "\n")
                    print("Results saved to: evaluation/results/stages_2_3_BETTER_MODEL.json")
                    print("\nView detailed results:")
                    print("  python view_detailed_results.py")
                    break
                
                # Check for errors
                if info['error']:
                    print("\n" + "="*80)
                    print("  ⚠️  ERROR DETECTED IN LOG")
                    print("="*80)
                    print("\nCheck the full log:")
                    print("  cat better_model_output.log")
                    break
                
                # Check if process stopped unexpectedly
                if not is_running and not info['evaluation_complete']:
                    print("\n" + "="*80)
                    print("  ⚠️  PROCESS STOPPED UNEXPECTEDLY")
                    print("="*80)
                    print("\nCheck the full log:")
                    print("  cat better_model_output.log")
                    break
            
            else:
                print("⏳ Waiting for log file to be created...")
            
            print(f"\n{'─'*80}")
            print("  Updating every 10 seconds... (Ctrl+C to stop monitoring)")
            print(f"{'─'*80}\n")
            
            # Wait before next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("  ⏸️  MONITORING STOPPED (Process still running in background)")
        print("="*80)
        print("\nTo resume monitoring:")
        print("  python monitor_progress.py")
        print("\nTo view full log:")
        print("  tail -f better_model_output.log")
        print()

if __name__ == "__main__":
    main()

