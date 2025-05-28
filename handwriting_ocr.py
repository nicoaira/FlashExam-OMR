import easyocr

def recognize_name_id(name_img_path, id_img_path, device='cpu'):
    reader = easyocr.Reader(['es', 'en'], gpu=(device=='cuda'))
    name_result = reader.readtext(name_img_path, detail=0, paragraph=True)
    id_result = reader.readtext(id_img_path, detail=0, paragraph=True)
    name_text = name_result[0] if name_result else ''
    id_text = id_result[0] if id_result else ''
    return name_text.strip(), id_text.strip()
