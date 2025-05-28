import os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_pdf(students_info_dir, output_pdf):
    # List all student subfolders
    student_dirs = [d for d in os.listdir(students_info_dir) if os.path.isdir(os.path.join(students_info_dir, d))]
    student_dirs.sort()
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    row_height = 80  # Fixed row height for compact layout
    img_disp_height = 60  # Display height in points (not pixels)
    img_disp_width = 220  # Display width in points (not pixels)
    font_size = 12
    c.setFont("Helvetica", font_size)
    # Prepare for CSV template
    csv_rows = []
    for student in student_dirs:
        student_path = os.path.join(students_info_dir, student)
        name_img_path = os.path.join(student_path, "name.png")
        id_img_path = os.path.join(student_path, "id.png")
        if not (os.path.exists(name_img_path) and os.path.exists(id_img_path)):
            continue
        # Draw filename
        c.drawString(margin, y + img_disp_height // 2 - font_size // 2, student)
        # Draw name image
        x_name = margin + 120
        try:
            name_img = Image.open(name_img_path)
            aspect = name_img.width / name_img.height
            # Fit to display box, but do not upscale
            scale = min(img_disp_width / name_img.width, img_disp_height / name_img.height, 1.0)
            draw_w = int(name_img.width * scale)
            draw_h = int(name_img.height * scale)
            # Center in box
            x_offset = x_name + (img_disp_width - draw_w) // 2
            y_offset = y + (img_disp_height - draw_h) // 2
            c.drawImage(ImageReader(name_img), x_offset, y_offset, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            pass
        # Draw id image
        x_id = x_name + img_disp_width + 40
        try:
            id_img = Image.open(id_img_path)
            aspect = id_img.width / id_img.height
            scale = min(img_disp_width / id_img.width, img_disp_height / id_img.height, 1.0)
            draw_w = int(id_img.width * scale)
            draw_h = int(id_img.height * scale)
            x_offset = x_id + (img_disp_width - draw_w) // 2
            y_offset = y + (img_disp_height - draw_h) // 2
            c.drawImage(ImageReader(id_img), x_offset, y_offset, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            pass
        csv_rows.append({"image": student, "name": "", "id": ""})
        y -= row_height
        if y < margin + row_height:
            c.showPage()
            c.setFont("Helvetica", font_size)
            y = height - margin
    c.save()
    print(f"PDF saved to {output_pdf}")
    # Write CSV template in the same directory as the PDF
    import csv
    output_dir = os.path.dirname(output_pdf)
    csv_path = os.path.join(output_dir, "image-to-name.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "name", "id"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV template saved to {csv_path}")

if __name__ == "__main__":
    import sys
    # Default to output/temaA/students-info/ and output/temaA/image-to-names.pdf
    students_info_dir = "output/temaA/students-info/"
    output_pdf = "output/temaA/image-to-names.pdf"
    if len(sys.argv) > 1:
        students_info_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_pdf = sys.argv[2]
    generate_pdf(students_info_dir, output_pdf)
