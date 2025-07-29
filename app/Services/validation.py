import re

# Valid codes
valid_state_codes = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HR",
    "HP", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


# Valid Indian Number Plate
def valid_indian_numb_plate_fn(text: str) -> bool:
    number_plate_regex = re.compile(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$')
    if text[0:2] not in valid_state_codes:
        return False
    if not text[-4].isdigit():
        return False
    return bool(number_plate_regex.match(text))


# Correction Function
def correct_plate_text(ocr_text: str) -> str:
    ocr_text = ocr_text.lstrip('F')
    
    corrections = {
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '7':'T', '4':'L',
        'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T':'7', 'L':'4', 'A':'4', 'D':'0'
    }
    
    cleaned = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
    corrected = ''
    
    for i, char in enumerate(cleaned):
        if i < 2: 
            corrected += corrections.get(char, char) if char in '012568' else char
        
        elif i < 4:
            if(i>2 and len(cleaned)==9):
                corrected += corrections.get(char, char) if char in '0125678' else char
            else:
                corrected += corrections.get(char, char) if char in 'AOIZSGBL' else char
                
        elif i < 7:
            if (i>5 and len(cleaned)==10):
                corrected += corrections.get(char, char) if char in 'ABDGILOSTZ' else char
            elif(i>4 and len(cleaned)==9):
                corrected += corrections.get(char, char) if char in 'ABDGILOSTZ' else char
            else:
                corrected += corrections.get(char, char) if char in '0125678' else char
        
        else:#7,8,9,10,11
            corrected += corrections.get(char, char) if char in 'ABDGILOSTZ' else char
    
    return corrected