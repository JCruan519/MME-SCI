import re

def get_message(sample, eval_mode='image_text_zh'):
    all_contents = []
    
    if 'image_text' in eval_mode:

        all_contents.append(sample['question'])
        if sample['options']:
            prefixes = ['A', 'B', 'C', 'D']
            for i, option in enumerate(sample['options']):
                if i < len(prefixes):
                    all_contents.append(f"{prefixes[i]}. {option}")

        message_content = "\n".join(all_contents)

        if sample['image_1']:
            all_contents = []
            
            image_fields = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
            split_text = re.split(r"<image_\d+>", message_content) 

            for i, fragment in enumerate(split_text):
                if fragment.strip(): 
                    all_contents.append({"type": "text", "text": fragment.strip()})
                if i < len(image_fields):
                    img_field = image_fields[i]
                    if sample[img_field]: 
                        img_base64 = sample[img_field]
                        all_contents.append({
                            "type": "image_url",
                            "image_url": {'url': f"data:image/png;base64,{img_base64}"}
                        })

        else:
            all_contents = [{"type": "text", "text": message_content}]

    elif eval_mode == 'image':
        img_base64 = sample['image']
        all_contents.append({
            "type": "image_url",
            "image_url": {'url': f"data:image/png;base64,{img_base64}"}
        })

    else:
        raise ValueError(f"eval_mode {eval_mode} not supported")

    return all_contents


def get_raw_question_and_options(sample):
    all_contents = []
    all_contents.append(sample['question'])
    if sample['options']:
        prefixes = ['A', 'B', 'C', 'D']
        for i, option in enumerate(sample['options']):
            if i < len(prefixes):
                all_contents.append(f"{prefixes[i]}. {option}")
    raw_question_and_options = " ".join(all_contents)

    return raw_question_and_options
