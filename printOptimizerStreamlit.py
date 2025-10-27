import streamlit as st
import random
from collections import Counter
import matplotlib.pyplot as plt
from rectpack import newPacker
import json
import csv
from io import BytesIO, StringIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches


# ==============================================================
# =============== OPTIMIZATION + VISUALIZATION =================
# ==============================================================

def try_packing(combination, sheet_size):
    """Try packing a given combination of rectangles using multiple algorithms."""
    import rectpack
    
    # Try ALL algorithm combinations to find the best packing
    pack_algorithms = [
        rectpack.MaxRectsBssf,
        rectpack.MaxRectsBaf,
        rectpack.MaxRectsBl,
        rectpack.MaxRectsBlsf,
    ]
    
    sort_algorithms = [
        rectpack.SORT_AREA,
        rectpack.SORT_DIFF,
        rectpack.SORT_RATIO,
        rectpack.SORT_NONE,
        rectpack.SORT_PERI,
        rectpack.SORT_SSIDE,
        rectpack.SORT_LSIDE,
    ]
    
    best_packer = None
    best_count = 0
    best_actual_height = float('inf')
    
    # Try all combinations
    for pack_algo in pack_algorithms:
        for sort_algo in sort_algorithms:
            packer = newPacker(rotation=True, pack_algo=pack_algo, sort_algo=sort_algo)
            for w, h in combination:
                packer.add_rect(w, h)
            packer.add_bin(sheet_size[0], sheet_size[1])
            packer.pack()
            
            count = len(packer.rect_list())
            
            # Calculate actual height used (max y+h position)
            actual_height = 0
            if count > 0:
                for b, x, y, w, h, rid in packer.rect_list():
                    actual_height = max(actual_height, y + h)
            
            # Prefer solutions that pack more items, or same items but use less height
            is_better = False
            if count > best_count:
                is_better = True
            elif count == best_count and actual_height < best_actual_height:
                is_better = True
                
            if is_better:
                best_count = count
                best_packer = packer
                best_actual_height = actual_height
    
    return best_packer, best_count, best_actual_height


def find_best_mix(rect_sizes, sheet_width, min_height, max_height, trials=1000):
    """Optimize mix of prints with flexible height constraints."""
    best_packer = None
    best_count = 0
    best_mix = None
    best_height = max_height if max_height > 0 else 500
    
    actual_min = min_height if min_height > 0 else 50
    actual_max = max_height if max_height > 0 else 500
    
    if min_height == 0 and max_height == 0:
        actual_min = 50
        actual_max = 500
    
    heights_to_try = []
    if max_height > 0:
        heights_to_try.append(max_height)
    if min_height > 0 and min_height != max_height:
        heights_to_try.append(min_height)

    for trial in range(trials):
        combination = []
        
        if trial < len(heights_to_try) * 2:
            for (w, h, min_required) in rect_sizes:
                combination.extend([(w, h)] * min_required)
        else:
            for (w, h, min_required) in rect_sizes:
                count = min_required + random.randint(0, 10)
                combination.extend([(w, h)] * count)

        if trial < len(heights_to_try):
            current_height = heights_to_try[trial]
        elif trial < len(heights_to_try) * 2:
            current_height = heights_to_try[trial - len(heights_to_try)]
        else:
            current_height = random.uniform(actual_min, actual_max)

        packer, packed_count, actual_height_used = try_packing(combination, (sheet_width, current_height))
        
        packed_rects = [(w, h) for b, x, y, w, h, rid in packer.rect_list()]
        meets_requirements = True
        for (w, h, min_required) in rect_sizes:
            packed_count_for_size = sum(1 for pw, ph in packed_rects if set([pw, ph]) == set([w, h]))
            if packed_count_for_size < min_required:
                meets_requirements = False
                break
        
        if meets_requirements and packed_count > best_count:
            best_count = packed_count
            best_packer = packer
            best_mix = combination
            best_height = actual_height_used

    return best_packer, best_count, best_mix, best_height


def plot_layout_web(packer, sheet_width, sheet_height, rect_sizes):
    """Draw the sheet layout using matplotlib - returns figure for Streamlit."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, sheet_width)
    ax.set_ylim(0, sheet_height)
    ax.set_aspect('equal')
    ax.set_title(f"Print Layout ({sheet_width:.1f} x {sheet_height:.1f} cm)")
    ax.set_xlabel("Width (cm)")
    ax.set_ylabel("Height (cm)")

    ax.add_patch(plt.Rectangle((0, 0), sheet_width, sheet_height,
                               fill=False, edgecolor='black', linewidth=2))

    base_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99',
                   '#c2c2f0', '#ffb3e6', '#a6ffcc', '#ffd699']
    color_map = {}
    for i, (w, h, _) in enumerate(rect_sizes):
        color_map[(w, h)] = base_colors[i % len(base_colors)]

    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        size_match = next((s for s in color_map if set(s[:2]) == set((w, h))), None)
        color = color_map.get(size_match, "#dddddd")

        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='black',
                                   facecolor=color, alpha=0.75))
        ax.text(x + w/2, y + h/2, f"{int(w)}x{int(h)}",
                ha='center', va='center', fontsize=8)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig


# ==============================================================
# ==================== EXPORT FUNCTIONS ========================
# ==============================================================

def export_to_pdf_with_cutmarks(packer, sheet_width, sheet_height, rect_sizes):
    """Generate a PDF with cut marks for the print shop."""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Create figure with proper size for printing
        fig = plt.figure(figsize=(11.69, 16.53))  # A3 size in inches
        ax = fig.add_subplot(111)
        
        ax.set_xlim(-5, sheet_width + 5)
        ax.set_ylim(-5, sheet_height + 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw sheet border
        ax.add_patch(plt.Rectangle((0, 0), sheet_width, sheet_height,
                                   fill=False, edgecolor='black', linewidth=2))
        
        # Add title and info
        ax.text(sheet_width/2, sheet_height + 3, 
                f"Print Layout - Sheet: {sheet_width:.1f} × {sheet_height:.1f} cm",
                ha='center', va='bottom', fontsize=14, weight='bold')
        
        # Color map for consistency
        base_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99',
                       '#c2c2f0', '#ffb3e6', '#a6ffcc', '#ffd699']
        color_map = {}
        for i, (w, h, _) in enumerate(rect_sizes):
            color_map[(w, h)] = base_colors[i % len(base_colors)]
        
        # Draw each print with cut marks
        for idx, rect in enumerate(packer.rect_list()):
            b, x, y, w, h, rid = rect
            size_match = next((s for s in color_map if set(s[:2]) == set((w, h))), None)
            color = color_map.get(size_match, "#dddddd")
            
            # Draw the print area
            ax.add_patch(plt.Rectangle((x, y), w, h, 
                                       edgecolor='black', 
                                       facecolor=color, 
                                       alpha=0.3,
                                       linewidth=1.5))
            
            # Add cut marks at corners (small L-shaped marks)
            mark_length = 0.5
            mark_offset = 0.2
            
            # Top-left corner
            ax.plot([x - mark_offset, x - mark_offset - mark_length], 
                   [y, y], 'k-', linewidth=0.5)
            ax.plot([x, x], 
                   [y - mark_offset, y - mark_offset - mark_length], 
                   'k-', linewidth=0.5)
            
            # Top-right corner
            ax.plot([x + w + mark_offset, x + w + mark_offset + mark_length], 
                   [y, y], 'k-', linewidth=0.5)
            ax.plot([x + w, x + w], 
                   [y - mark_offset, y - mark_offset - mark_length], 
                   'k-', linewidth=0.5)
            
            # Bottom-left corner
            ax.plot([x - mark_offset, x - mark_offset - mark_length], 
                   [y + h, y + h], 'k-', linewidth=0.5)
            ax.plot([x, x], 
                   [y + h + mark_offset, y + h + mark_offset + mark_length], 
                   'k-', linewidth=0.5)
            
            # Bottom-right corner
            ax.plot([x + w + mark_offset, x + w + mark_offset + mark_length], 
                   [y + h, y + h], 'k-', linewidth=0.5)
            ax.plot([x + w, x + w], 
                   [y + h + mark_offset, y + h + mark_offset + mark_length], 
                   'k-', linewidth=0.5)
            
            # Label with dimensions and ID
            ax.text(x + w/2, y + h/2, 
                   f"#{idx+1}\n{int(w)}×{int(h)} cm",
                   ha='center', va='center', 
                   fontsize=8, weight='bold')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
    
    buffer.seek(0)
    return buffer


def export_to_csv(packer, sheet_width, sheet_height):
    """Generate a CSV file with print positions and dimensions."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Print_ID', 'Width_cm', 'Height_cm', 'Position_X_cm', 'Position_Y_cm', 
                     'Top_Left_X', 'Top_Left_Y', 'Bottom_Right_X', 'Bottom_Right_Y'])
    
    # Write each rectangle
    for idx, rect in enumerate(packer.rect_list()):
        b, x, y, w, h, rid = rect
        writer.writerow([
            idx + 1,
            f"{w:.2f}",
            f"{h:.2f}",
            f"{x:.2f}",
            f"{y:.2f}",
            f"{x:.2f}",
            f"{y:.2f}",
            f"{x + w:.2f}",
            f"{y + h:.2f}"
        ])
    
    # Summary at the end
    writer.writerow([])
    writer.writerow(['SUMMARY'])
    writer.writerow(['Sheet Width (cm)', f"{sheet_width:.2f}"])
    writer.writerow(['Sheet Height (cm)', f"{sheet_height:.2f}"])
    writer.writerow(['Total Prints', len(packer.rect_list())])
    
    output.seek(0)
    return output.getvalue()


def export_to_json(packer, sheet_width, sheet_height, rect_sizes):
    """Generate a JSON file with complete layout information."""
    layout_data = {
        'sheet': {
            'width_cm': sheet_width,
            'height_cm': sheet_height
        },
        'prints': []
    }
    
    for idx, rect in enumerate(packer.rect_list()):
        b, x, y, w, h, rid = rect
        layout_data['prints'].append({
            'id': idx + 1,
            'width_cm': w,
            'height_cm': h,
            'position': {
                'x_cm': x,
                'y_cm': y
            },
            'corners': {
                'top_left': {'x': x, 'y': y},
                'top_right': {'x': x + w, 'y': y},
                'bottom_left': {'x': x, 'y': y + h},
                'bottom_right': {'x': x + w, 'y': y + h}
            }
        })
    
    # Count by size
    packed_rects = [(w, h) for b, x, y, w, h, rid in packer.rect_list()]
    counts = Counter(packed_rects)
    
    layout_data['summary'] = {
        'total_prints': len(packer.rect_list()),
        'by_size': [
            {'width_cm': w, 'height_cm': h, 'count': c}
            for (w, h), c in counts.items()
        ]
    }
    
    return json.dumps(layout_data, indent=2)


# ==============================================================
# ==================== STREAMLIT APP ===========================
# ==============================================================

def initialize_session_state():
    """Initialize session state with default values (per-user storage)."""
    if 'sheet_width' not in st.session_state:
        st.session_state.sheet_width = 103.0
    if 'min_height' not in st.session_state:
        st.session_state.min_height = 0.0
    if 'max_height' not in st.session_state:
        st.session_state.max_height = 150.0
    if 'rectangles' not in st.session_state:
        st.session_state.rectangles = []
    if 'last_optimization' not in st.session_state:
        st.session_state.last_optimization = None


def main():
    st.set_page_config(page_title="Print Layout Optimizer", layout="wide", page_icon="🖼️")
    
    st.title("🖼️ Print Layout Optimizer")
    st.markdown("---")
    
    # Initialize per-user session state
    initialize_session_state()
    
    # Sheet settings
    st.header("📏 Sheet Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sheet_width = st.number_input(
            "Width (cm)", 
            value=st.session_state.sheet_width,
            min_value=1.0,
            step=0.1,
            key="input_sheet_width"
        )
        st.session_state.sheet_width = sheet_width
    
    with col2:
        min_height = st.number_input(
            "Min Height (cm, 0=no limit)", 
            value=st.session_state.min_height,
            min_value=0.0,
            step=0.1,
            key="input_min_height"
        )
        st.session_state.min_height = min_height
    
    with col3:
        max_height = st.number_input(
            "Max Height (cm, 0=no limit)", 
            value=st.session_state.max_height,
            min_value=0.0,
            step=0.1,
            key="input_max_height"
        )
        st.session_state.max_height = max_height
    
    st.markdown("---")
    
    # Add print sizes
    st.header("📐 Add Print Size")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        new_width = st.number_input("Width (cm)", value=42.0, min_value=1.0, step=0.1, key="new_w")
    
    with col2:
        new_height = st.number_input("Height (cm)", value=60.0, min_value=1.0, step=0.1, key="new_h")
    
    with col3:
        new_min = st.number_input("Min Required", value=1, min_value=0, step=1, key="new_min")
    
    with col4:
        st.write("")
        st.write("")
        if st.button("➕ Add", use_container_width=True):
            st.session_state.rectangles.append([new_width, new_height, new_min])
            st.rerun()
    
    st.markdown("---")
    
    # Display current rectangles
    if st.session_state.rectangles:
        st.header("📦 Current Print Sizes")
        
        for i, (w, h, min_req) in enumerate(st.session_state.rectangles):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(f"📐 {w} × {h} cm (minimum {min_req} piece{'s' if min_req != 1 else ''})")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.rectangles.pop(i)
                    st.rerun()
        
        if st.button("🗑️ Delete All", type="secondary"):
            st.session_state.rectangles = []
            st.rerun()
        
        st.markdown("---")
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("🔄 Running optimization... This may take a moment."):
                rect_tuples = [tuple(r) for r in st.session_state.rectangles]
                packer, total, mix, best_h = find_best_mix(
                    rect_tuples, 
                    sheet_width, 
                    min_height, 
                    max_height, 
                    trials=2000
                )
                
                if packer:
                    # Store optimization result in session state (per-user)
                    st.session_state.last_optimization = {
                        'packer': packer,
                        'total': total,
                        'best_h': best_h,
                        'sheet_width': sheet_width,
                        'rect_sizes': rect_tuples
                    }
                    
                    st.success(f"✅ Optimization Complete!")
                    
                    # Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Prints", total)
                    with col2:
                        st.metric("Sheet Height Used", f"{best_h:.1f} cm")
                    
                    # Count what was actually packed
                    packed_rects = [(w, h) for b, x, y, w, h, rid in packer.rect_list()]
                    counts = Counter(packed_rects)
                    
                    st.subheader("📊 Combination:")
                    for (w, h), c in sorted(counts.items(), key=lambda x: -x[1]):
                        st.write(f"• **{w} × {h} cm**: {c} piece{'s' if c != 1 else ''}")
                    
                    # Plot
                    st.subheader("🎨 Layout Visualization")
                    fig = plot_layout_web(packer, sheet_width, best_h, rect_tuples)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("❌ Optimization failed. No valid layout found within the constraints.")
                    st.info("Try adjusting the height limits or reducing minimum requirements.")
        
        # Export section (only show if optimization was run)
        if st.session_state.last_optimization:
            st.markdown("---")
            st.header("📥 Export for Print Shop")
            st.write("Download the layout in different formats for your printer:")
            
            opt = st.session_state.last_optimization
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # PDF with cut marks
                pdf_buffer = export_to_pdf_with_cutmarks(
                    opt['packer'], 
                    opt['sheet_width'], 
                    opt['best_h'], 
                    opt['rect_sizes']
                )
                st.download_button(
                    label="📄 Download PDF (Cut Marks)",
                    data=pdf_buffer,
                    file_name="print_layout_cutmarks.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                # CSV
                csv_data = export_to_csv(
                    opt['packer'], 
                    opt['sheet_width'], 
                    opt['best_h']
                )
                st.download_button(
                    label="📊 Download CSV (Positions)",
                    data=csv_data,
                    file_name="print_layout.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # JSON
                json_data = export_to_json(
                    opt['packer'], 
                    opt['sheet_width'], 
                    opt['best_h'], 
                    opt['rect_sizes']
                )
                st.download_button(
                    label="🔧 Download JSON (Complete)",
                    data=json_data,
                    file_name="print_layout.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.info("💡 **Tip:** The PDF includes cut marks for precise cutting. The CSV and JSON files contain exact coordinates for automated cutting systems.")
            
    else:
        st.info("👆 Please add at least one print size to begin optimization.")


if __name__ == "__main__":
    main()