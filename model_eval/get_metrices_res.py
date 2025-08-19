import json
import argparse

parser = argparse.ArgumentParser(description="Script to calculate accuracy for each course based on JSONL file.")
parser.add_argument("--JUDGE_MODEL", type=str, default="xxx")
parser.add_argument("--test_model_name", type=str, default="xxx")
parser.add_argument("--eval_mode", type=str, default='xxx')
parser.add_argument("--test_data_name", type=str, default="xxx")
args = parser.parse_args()

JUDGE_MODEL = args.JUDGE_MODEL
test_model_name = args.test_model_name
test_data_name = args.test_data_name
eval_mode = args.eval_mode

file_path = f'../eval_res-{eval_mode}/{test_data_name}/{test_model_name}_judged_by_{JUDGE_MODEL}.jsonl'

course_stats = {}
total_stats = {'total': 0, 'correct': 0}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line.strip())
        
        course = sample.get('course')
        judge_answer = sample.get('judge_answer')
        
        if course is not None and judge_answer is not None:
            if course not in course_stats:
                course_stats[course] = {'total': 0, 'correct': 0}
            
            course_stats[course]['total'] += 1
            if judge_answer == 1:
                course_stats[course]['correct'] += 1
            
            total_stats['total'] += 1
            if judge_answer == 1:
                total_stats['correct'] += 1

course_accuracy = {}
for course, stats in course_stats.items():
    total = stats['total']
    correct = stats['correct']
    accuracy = correct / total * 100 if total > 0 else 0 
    course_accuracy[course] = accuracy

overall_accuracy = total_stats['correct'] / total_stats['total'] * 100 if total_stats['total'] > 0 else 0 

print("=" * 30) 
print(f"total questions: {total_stats['total']}")
for course, accuracy in course_accuracy.items():
    print(f"Course: {course}, Accuracy: {accuracy:.2f}%") 
print(f"Overall accuracy: {overall_accuracy:.2f}%") 
print("=" * 30)