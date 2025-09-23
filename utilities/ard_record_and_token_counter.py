import json
import os
import time
import tiktoken

def extract_all_text(record):
    """Extract all text content from a JSON record"""
    def extract_strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from extract_strings(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from extract_strings(item)
    
    return ' '.join(extract_strings(record))

def safe_encode_text(encoder, text):
    """Safely encode text, handling special tokens"""
    try:
        # Allow special tokens to be encoded normally
        return len(encoder.encode(text, disallowed_special=()))
    except Exception as e:
        # Fallback: estimate tokens as chars/4 if encoding fails
        return len(text) // 4

def count_tokens_tiktoken():
    """Count tokens using only tiktoken tokenizer"""
    
    print("Loading tiktoken encoder...")
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    if not jsonl_files:
        print("No JSONL files found")
        return
    
    # Show file overview
    total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in jsonl_files)
    print(f"\nFound {len(jsonl_files)} JSONL files ({total_size_mb:.1f} MB total)")
    
    print("\n" + "=" * 70)
    print("TOKEN COUNTING WITH TIKTOKEN")
    print("=" * 70)
    
    total_tokens = 0
    total_records = 0
    failed_records = 0
    file_results = []  # Store results for each file
    overall_start = time.time()
    
    for i, file in enumerate(sorted(jsonl_files)):
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"\n[{i+1}/{len(jsonl_files)}] Processing {file} ({file_size_mb:.1f} MB)...")
        
        file_tokens = 0
        file_records = 0
        file_failed = 0
        file_start = time.time()
        last_update = time.time()
        
        try:
            file_size_bytes = os.path.getsize(file)
            bytes_processed = 0
            
            with open(file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            text = extract_all_text(record)
                            
                            # Safe encoding with special token handling
                            tokens = safe_encode_text(encoder, text)
                            file_tokens += tokens
                            file_records += 1
                            
                            # Track progress
                            bytes_processed += len(line.encode('utf-8'))
                            current_time = time.time()
                            
                            # Update every 2 seconds or every 500 records
                            if (file_records % 500 == 0 or 
                                current_time - last_update > 2.0):
                                
                                progress_pct = min(100, (bytes_processed / file_size_bytes) * 100)
                                elapsed = current_time - file_start
                                rate = file_records / elapsed if elapsed > 0 else 0
                                
                                print(f"    Progress: {progress_pct:5.1f}% | "
                                      f"{file_records:,} records | "
                                      f"{file_tokens:,} tokens | "
                                      f"{rate:.0f} rec/sec | "
                                      f"{elapsed:.1f}s elapsed")
                                
                                last_update = current_time
                                
                        except json.JSONDecodeError:
                            file_failed += 1
                            continue
                        except Exception as e:
                            file_failed += 1
                            continue
            
            file_elapsed = time.time() - file_start
            success_rate = (file_records / (file_records + file_failed)) * 100 if (file_records + file_failed) > 0 else 0
            
            print(f"    ✓ Completed: {file_tokens:,} tokens ({file_records:,} records) in {file_elapsed:.1f}s")
            if file_failed > 0:
                print(f"    ⚠ Failed to process {file_failed} records ({success_rate:.1f}% success rate)")
            
            total_tokens += file_tokens
            total_records += file_records
            failed_records += file_failed
            
            # Store file results
            file_results.append({
                'file': file,
                'source': file.replace('.jsonl', ''),
                'tokens': file_tokens,
                'records': file_records,
                'failed': file_failed,
                'avg_tokens': file_tokens / file_records if file_records > 0 else 0,
                'time': file_elapsed
            })
            
        except Exception as e:
            print(f"    ✗ Error processing file: {e}")
            file_results.append({
                'file': file,
                'source': file.replace('.jsonl', ''),
                'tokens': 0,
                'records': 0,
                'failed': 0,
                'avg_tokens': 0,
                'time': 0
            })
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print("RESULTS BY DATA SOURCE")
    print("=" * 70)
    print(f"{'Source':<20} {'Records':<10} {'Tokens':<15} {'Avg/Record':<12} {'% of Total':<10}")
    print("-" * 70)
    
    # Sort by token count (descending)
    sorted_results = sorted(file_results, key=lambda x: x['tokens'], reverse=True)
    
    for result in sorted_results:
        source = result['source']
        records = result['records']
        tokens = result['tokens']
        avg_tokens = result['avg_tokens']
        pct_total = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"{source:<20} {records:<10,} {tokens:<15,} {avg_tokens:<12.1f} {pct_total:<10.1f}%")
        
        if result['failed'] > 0:
            print(f"{'  (failed: ' + str(result['failed']) + ')':<20}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_records:<10,} {total_tokens:<15,} {total_tokens/total_records:<12.1f} {'100.0%':<10}")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total tokens:           {total_tokens:,}")
    print(f"Total records:          {total_records:,}")
    print(f"Failed records:         {failed_records:,}")
    print(f"Success rate:           {(total_records/(total_records+failed_records))*100:.1f}%")
    print(f"Average tokens/record:  {total_tokens/total_records:.1f}")
    print(f"Total processing time:  {overall_elapsed:.1f} seconds")
    print(f"Overall rate:           {total_records/overall_elapsed:.0f} records/second")
    
    # Size estimates
    print(f"\nData size estimates:")
    print(f"  At $0.0015/1K tokens:  ${total_tokens * 0.0015 / 1000:.2f}")
    print(f"  At $0.002/1K tokens:   ${total_tokens * 0.002 / 1000:.2f}")

if __name__ == "__main__":
    print("Tiktoken Token Counter")
    print("Note: This will handle special tokens safely")
    print("-" * 50)
    
    count_tokens_tiktoken()