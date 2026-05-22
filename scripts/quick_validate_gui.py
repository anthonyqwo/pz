from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from puzzle_recognition.calibration import parse_corners
from puzzle_recognition.config import BOARDS_DIR, DetectorConfig, MatcherConfig, OUTPUTS_DIR
from puzzle_recognition.io_utils import read_json
from puzzle_recognition.piece_labeler import label_pieces
from puzzle_recognition.recognizer import recognize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_boards() -> list[str]:
    if not BOARDS_DIR.exists():
        return []
    boards = []
    for path in sorted(BOARDS_DIR.iterdir()):
        if path.is_dir() and (path / "board_config.json").exists():
            boards.append(path.name)
    return boards


def make_preview_image(source_path: str | Path, max_width: int = 980, max_height: int = 620) -> Path:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read preview image: {source_path}")

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    preview_dir = OUTPUTS_DIR / "gui_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "preview.png"
    cv2.imwrite(str(preview_path), image)
    return preview_path


def make_slot_preview_image(source_path: str | Path, max_width: int = 250, max_height: int = 250) -> Path:
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read slot mask image: {source_path}")

    # Crop to the slot bounding box to zoom in and display shape details beautifully
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = 12
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(image.shape[0], y + h + pad)
        x1 = min(image.shape[1], x + w + pad)
        cropped = image[y0:y1, x0:x1]
    else:
        cropped = image

    # Add a thin white border around the cropped mask for visual contrast
    cropped = cv2.copyMakeBorder(cropped, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=60)

    height, width = cropped.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        cropped = cv2.resize(cropped, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    preview_dir = OUTPUTS_DIR / "gui_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "slot_preview.png"
    cv2.imwrite(str(preview_path), cropped)
    return preview_path


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)


class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.is_collapsed = True
        
        self.toggle_btn = ttk.Button(
            self,
            text=f"➕ {self.title} (點擊展開)",
            command=self.toggle,
            style="Collapsible.TButton"
        )
        self.toggle_btn.pack(fill=tk.X, expand=True)
        
        self.sub_frame = ttk.Frame(self, padding=(10, 5))
        
    def toggle(self):
        if self.is_collapsed:
            self.sub_frame.pack(fill=tk.X, expand=True, pady=(4, 0))
            self.toggle_btn.configure(text=f"➖ {self.title} (點擊收摺)")
            self.is_collapsed = False
        else:
            self.sub_frame.pack_forget()
            self.toggle_btn.configure(text=f"➕ {self.title} (點擊展開)")
            self.is_collapsed = True


class QuickValidateApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Puzzle Recognition Quick Validator")
        self.root.geometry("1240x880")

        self.image_path = tk.StringVar()
        self.board_id = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(PROJECT_ROOT / "outputs" / "quick_validate"))
        self.already_rectified = tk.BooleanVar(value=True)
        self.corners = tk.StringVar()
        
        # High resolution presets
        self.gray_threshold = tk.IntVar(value=60)
        self.min_piece_area = tk.DoubleVar(value=800.0)
        self.max_piece_area = tk.DoubleVar(value=350000.0)
        self.morphology_kernel_size = tk.IntVar(value=9)
        self.min_bbox_width = tk.IntVar(value=20)
        self.min_bbox_height = tk.IntVar(value=20)
        self.max_bbox_width = tk.IntVar(value=800)
        self.max_bbox_height = tk.IntVar(value=800)
        self.aspect_ratio_min = tk.DoubleVar(value=0.35)
        self.aspect_ratio_max = tk.DoubleVar(value=2.8)
        self.extent_min = tk.DoubleVar(value=0.20)
        self.extent_max = tk.DoubleVar(value=0.90)
        self.min_solidity = tk.DoubleVar(value=0.35)
        self.max_solidity = tk.DoubleVar(value=1.0)
        self.max_piece_mean_l = tk.DoubleVar(value=80.0)
        self.border_margin = tk.IntVar(value=8)
        self.reject_border_components = tk.BooleanVar(value=True)
        self.expected_max_pieces = tk.IntVar(value=10)
        self.close_iterations = tk.IntVar(value=2)
        self.contour_epsilon_ratio = tk.DoubleVar(value=0.002)
        self.debug_label_min_area = tk.DoubleVar(value=800.0)
        
        self.allow_mirror = tk.BooleanVar(value=False)
        self.save_crops = tk.BooleanVar(value=True)
        self.label_preview_output = tk.StringVar(value="original_numbered_path")
        self.status = tk.StringVar(value="選擇圖片與底板後開始驗證")
        self.preview_image: tk.PhotoImage | None = None
        self.current_payload: dict | None = None
        self.piece_crop_image: tk.PhotoImage | None = None
        self.slot_preview_image: tk.PhotoImage | None = None
        self.selected_slot_path: Path | None = None

        boards = discover_boards()
        if boards:
            self.board_id.set(boards[0])

        self._build_layout(boards)

    def _build_layout(self, boards: list[str]) -> None:
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure modern aesthetic styling and colors
        style.configure('.', font=('Microsoft JhengHei', 10))
        style.configure('TLabel', font=('Microsoft JhengHei', 10))
        style.configure('TLabelframe', font=('Microsoft JhengHei', 10, 'bold'))
        style.configure('Header.TLabel', font=('Microsoft JhengHei', 12, 'bold'), foreground='#1a73e8')
        style.configure('Collapsible.TButton', font=('Microsoft JhengHei', 10, 'bold'), anchor='w')
        
        # Modern execution buttons
        style.configure('RunLabel.TButton', font=('Microsoft JhengHei', 11, 'bold'), background='#4caf50', foreground='white')
        style.map('RunLabel.TButton', background=[('active', '#45a049')])
        
        style.configure('RunRecognize.TButton', font=('Microsoft JhengHei', 11, 'bold'), background='#2196f3', foreground='white')
        style.map('RunRecognize.TButton', background=[('active', '#1e88e5')])

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Left Column (Sidebar outer container, fixed width)
        sidebar_outer = ttk.Frame(main, width=380, padding=(0, 0, 5, 0))
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        sidebar_outer.pack_propagate(False)

        sidebar_scroll = ScrollableFrame(sidebar_outer)
        sidebar_scroll.pack(fill=tk.BOTH, expand=True)
        sidebar = sidebar_scroll.scrollable_frame

        # Right Column (Main content pane)
        main_panel = ttk.Frame(main, padding=(5, 0, 0, 0))
        main_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left Sidebar Step 1: Input Setup
        input_frame = ttk.LabelFrame(sidebar, text="【步驟 1：輸入設定】", padding=8)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="待分類圖片:").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(input_frame, textvariable=self.image_path).grid(row=0, column=1, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(input_frame, text="選擇", command=self.choose_image, width=6).grid(row=0, column=2, pady=4)

        ttk.Label(input_frame, text="底板編號:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(input_frame, textvariable=self.board_id, values=boards, state="readonly").grid(row=1, column=1, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(input_frame, text="整理", command=self.refresh_boards, width=6).grid(row=1, column=2, pady=4)

        ttk.Label(input_frame, text="輸出目錄:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(input_frame, textvariable=self.output_dir).grid(row=2, column=1, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(input_frame, text="選擇", command=self.choose_output_dir, width=6).grid(row=2, column=2, pady=4)

        ttk.Checkbutton(input_frame, text="圖片已經是校正後俯視圖", variable=self.already_rectified).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=4)

        ttk.Label(input_frame, text="四角座標:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Entry(input_frame, textvariable=self.corners).grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=(4, 0), pady=4)
        ttk.Label(input_frame, text="格式: x1,y1;x2,y2;x3,y3;x4,y4", font=('Microsoft JhengHei', 8, 'italic'), foreground='gray').grid(row=5, column=0, columnspan=3, sticky=tk.W)

        # Left Sidebar Step 2: Actions & Execution
        actions_frame = ttk.LabelFrame(sidebar, text="【步驟 2：核心操作】", padding=8)
        actions_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(actions_frame, text="🟢 只偵測並編號拼圖 (Label Only)", command=self.run_label, style="RunLabel.TButton").pack(fill=tk.X, pady=6, ipady=4)
        ttk.Button(actions_frame, text="🔵 執行底板匹配 (Match Pieces)", command=self.run_recognize, style="RunRecognize.TButton").pack(fill=tk.X, pady=6, ipady=4)

        aux_frame = ttk.Frame(actions_frame)
        aux_frame.pack(fill=tk.X, pady=4)
        ttk.Button(aux_frame, text="大圖預設參數", command=self.apply_large_photo_preset).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(aux_frame, text="📂 打開輸出", command=self.open_output_dir).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        status_frame = ttk.Frame(actions_frame, relief=tk.SOLID, borderwidth=1)
        status_frame.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(status_frame, text="狀態監控:", font=('Microsoft JhengHei', 9, 'bold')).pack(anchor=tk.W, padx=6, pady=(4, 2))
        self.status_lbl = ttk.Label(status_frame, textvariable=self.status, font=('Microsoft JhengHei', 9), foreground='#d32f2f', wraplength=320)
        self.status_lbl.pack(anchor=tk.W, padx=6, pady=(0, 4))

        # Left Sidebar Step 3: Collapsible Parameters
        self.params_collapsible = CollapsibleFrame(sidebar, title="進階參數調校 (Power Tuning)")
        self.params_collapsible.pack(fill=tk.X, pady=(0, 10))

        params_sub = self.params_collapsible.sub_frame
        params_sub.columnconfigure(1, weight=1)
        params_sub.columnconfigure(3, weight=1)

        self._spin(params_sub, "gray thresh", self.gray_threshold, 0, 0, 0, 255, 1)
        self._spin(params_sub, "kernel size", self.morphology_kernel_size, 0, 2, 1, 99, 2)
        
        self._spin(params_sub, "min area", self.min_piece_area, 1, 0, 0, 10_000_000, 100)
        self._spin(params_sub, "max area", self.max_piece_area, 1, 2, 0, 50_000_000, 1000)
        
        self._spin(params_sub, "min bbox w", self.min_bbox_width, 2, 0, 0, 5000, 10)
        self._spin(params_sub, "min bbox h", self.min_bbox_height, 2, 2, 0, 5000, 10)
        
        self._spin(params_sub, "max bbox w", self.max_bbox_width, 3, 0, 0, 5000, 10)
        self._spin(params_sub, "max bbox h", self.max_bbox_height, 3, 2, 0, 5000, 10)
        
        self._spin(params_sub, "aspect min", self.aspect_ratio_min, 4, 0, 0, 20, 0.05)
        self._spin(params_sub, "aspect max", self.aspect_ratio_max, 4, 2, 0, 20, 0.05)
        
        self._spin(params_sub, "extent min", self.extent_min, 5, 0, 0, 1, 0.05)
        self._spin(params_sub, "extent max", self.extent_max, 5, 2, 0, 1, 0.05)
        
        self._spin(params_sub, "solidity min", self.min_solidity, 6, 0, 0, 1, 0.05)
        self._spin(params_sub, "solidity max", self.max_solidity, 6, 2, 0, 1, 0.01)
        
        self._spin(params_sub, "max mean L", self.max_piece_mean_l, 7, 0, 0, 255, 1)
        self._spin(params_sub, "border marg", self.border_margin, 7, 2, 0, 100, 1)
        
        self._spin(params_sub, "expected max", self.expected_max_pieces, 8, 0, 1, 1000, 1)
        self._spin(params_sub, "close iter", self.close_iterations, 8, 2, 1, 10, 1)
        
        self._spin(params_sub, "epsilon", self.contour_epsilon_ratio, 9, 0, 0, 0.05, 0.001)
        self._spin(params_sub, "debug area", self.debug_label_min_area, 9, 2, 0, 100000, 100)
        
        ttk.Checkbutton(params_sub, text="reject border", variable=self.reject_border_components).grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=4)
        ttk.Checkbutton(params_sub, text="允許鏡像", variable=self.allow_mirror).grid(row=10, column=2, columnspan=2, sticky=tk.W, pady=4)
        ttk.Checkbutton(params_sub, text="輸出每片 crop", variable=self.save_crops).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=4)
        
        ttk.Label(params_sub, text="標示預覽:").grid(row=12, column=0, sticky=tk.W, pady=4)
        preview_combo = ttk.Combobox(
            params_sub,
            textvariable=self.label_preview_output,
            values=[
                "original_numbered_path",
                "kept_pieces_overlay_path",
                "rejected_components_overlay_path",
                "all_components_overlay_path",
                "pieces_contours_path",
            ],
            state="readonly",
            width=18
        )
        preview_combo.grid(row=12, column=1, columnspan=3, sticky=tk.EW, padx=(4, 0), pady=4)

        # Right Main Panel Layout (Horizontal PanedWindow)
        result_area = ttk.PanedWindow(main_panel, orient=tk.HORIZONTAL)
        result_area.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Full Image Visual Preview
        preview_frame = ttk.LabelFrame(result_area, text="圖片標示主視覺預覽 (Major Preview)", padding=8)
        self.preview_label = ttk.Label(preview_frame, anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        result_area.add(preview_frame, weight=5)

        # Right Pane: Results and Dual-View Comparison Dashboard
        details_pane = ttk.Frame(result_area, padding=4)
        result_area.add(details_pane, weight=4)

        # Upper Half: Results Detail Table (Treeview)
        table_frame = ttk.LabelFrame(details_pane, text="辨識與匹配列表 (Piece & Slot List)", padding=6)
        table_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0, 6))

        tree_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("piece_id", "matched_slot_id", "confidence", "status", "offset"),
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
            selectmode="browse"
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y.config(command=self.tree.yview)
        tree_scroll_x.config(command=self.tree.xview)

        self.tree.heading("piece_id", text="拼圖編號")
        self.tree.heading("matched_slot_id", text="匹配底板編號")
        self.tree.heading("confidence", text="信心度")
        self.tree.heading("status", text="狀態")
        self.tree.heading("offset", text="偏移量(dx,dy)")

        self.tree.column("piece_id", width=70, anchor=tk.CENTER)
        self.tree.column("matched_slot_id", width=95, anchor=tk.CENTER)
        self.tree.column("confidence", width=70, anchor=tk.CENTER)
        self.tree.column("status", width=75, anchor=tk.CENTER)
        self.tree.column("offset", width=80, anchor=tk.CENTER)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", lambda e: self.popup_slot_mask())

        # Lower Half: Premium Dual-View Comparison Dashboard
        dual_view_frame = ttk.LabelFrame(details_pane, text="🧩 雙向比對儀表板 (Dual-View Dashboard)", padding=8)
        dual_view_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

        # 3 horizontal columns inside dual_view_frame
        # Column 1: Piece photo crop (on-the-fly)
        piece_preview_frame = ttk.LabelFrame(dual_view_frame, text="🧩 拼圖局部切片 (Actual Piece)", padding=4)
        piece_preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        
        self.piece_preview_label = ttk.Label(piece_preview_frame, anchor=tk.CENTER, text="等待定位偵測", background="#e1e1e1", width=22)
        self.piece_preview_label.pack(fill=tk.BOTH, expand=True, ipady=30)

        # Column 2: Slot mask crop
        slot_preview_frame = ttk.LabelFrame(dual_view_frame, text="🕳️ 匹配底板孔位 (Slot Mask)", padding=4)
        slot_preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        
        self.slot_preview_label = ttk.Label(slot_preview_frame, anchor=tk.CENTER, text="選擇拼圖以檢視孔位", background="#e1e1e1", width=22)
        self.slot_preview_label.pack(fill=tk.BOTH, expand=True, ipady=30)
        
        self.popup_slot_btn = ttk.Button(slot_preview_frame, text="🔍 彈出完整遮罩", command=self.popup_slot_btn_clicked)
        self.popup_slot_btn.pack(fill=tk.X, pady=(4, 0))
        self.popup_slot_btn.config(state="disabled")

        # Column 3: Detailed numeric attributes & JSON text view
        info_pane = ttk.LabelFrame(dual_view_frame, text="📋 詳細匹配與幾何數據", padding=4)
        info_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        text_scroll_y = ttk.Scrollbar(info_pane, orient=tk.VERTICAL)
        text_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.detail_text = tk.Text(info_pane, wrap=tk.WORD, width=28, height=8, yscrollcommand=text_scroll_y.set, font=('Consolas', 9))
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        text_scroll_y.config(command=self.detail_text.yview)

    def _spin(self, parent: ttk.Frame, label: str, variable: tk.Variable, row: int, column: int, from_: float, to: float, increment: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(0, 4), pady=4)
        ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, increment=increment, width=10).grid(row=row, column=column + 1, sticky=tk.W, padx=(0, 12), pady=4)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇待分類圖片",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if path:
            self.image_path.set(path)
            self.show_preview(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="選擇輸出資料夾")
        if path:
            self.output_dir.set(path)

    def refresh_boards(self) -> None:
        boards = discover_boards()
        self.status.set(f"找到 {len(boards)} 個底板")
        if boards and self.board_id.get() not in boards:
            self.board_id.set(boards[0])

    def apply_large_photo_preset(self) -> None:
        self.gray_threshold.set(60)
        self.max_piece_area.set(350000.0)
        self.max_bbox_width.set(800)
        self.max_bbox_height.set(800)
        self.max_solidity.set(1.0)
        self.expected_max_pieces.set(10)
        self.label_preview_output.set("original_numbered_path")
        self.status.set("已載入大圖高解析最佳化預設參數")

    def open_output_dir(self) -> None:
        path = Path(self.output_dir.get())
        path.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(path)
        except OSError as exc:
            messagebox.showerror("無法開啟資料夾", str(exc))

    def detector_config(self) -> DetectorConfig:
        return DetectorConfig(
            gray_threshold=int(self.gray_threshold.get()),
            min_piece_area=float(self.min_piece_area.get()),
            max_piece_area=float(self.max_piece_area.get()),
            morphology_kernel_size=int(self.morphology_kernel_size.get()),
            min_bbox_width=int(self.min_bbox_width.get()),
            min_bbox_height=int(self.min_bbox_height.get()),
            max_bbox_width=int(self.max_bbox_width.get()),
            max_bbox_height=int(self.max_bbox_height.get()),
            aspect_ratio_min=float(self.aspect_ratio_min.get()),
            aspect_ratio_max=float(self.aspect_ratio_max.get()),
            extent_min=float(self.extent_min.get()),
            extent_max=float(self.extent_max.get()),
            solidity_min=float(self.min_solidity.get()),
            solidity_max=float(self.max_solidity.get()),
            max_piece_mean_L=float(self.max_piece_mean_l.get()),
            border_margin=int(self.border_margin.get()),
            reject_border_components=bool(self.reject_border_components.get()),
            expected_max_pieces=int(self.expected_max_pieces.get()),
            close_iterations=int(self.close_iterations.get()),
            contour_epsilon_ratio=float(self.contour_epsilon_ratio.get()),
            debug_label_min_area=float(self.debug_label_min_area.get()),
        )

    def validate_image(self) -> Path:
        path = Path(self.image_path.get())
        if not path.exists():
            raise FileNotFoundError("請先選擇待分類圖片")
        return path

    def validate_board(self) -> str:
        board = self.board_id.get().strip()
        if not board:
            raise ValueError("請先選擇底板")
        if not (BOARDS_DIR / board / "board_config.json").exists():
            raise FileNotFoundError(f"找不到底板設定：{BOARDS_DIR / board / 'board_config.json'}")
        return board

    def run_label(self) -> None:
        try:
            image_path = self.validate_image()
            output_dir = Path(self.output_dir.get()) / "label"
            config = self.detector_config()
            save_crops = bool(self.save_crops.get())
            preview_key = self.label_preview_output.get()
            self.run_background_task(
                target=self._run_label_worker,
                args=(image_path, output_dir, config, save_crops, preview_key)
            )
        except Exception as exc:
            self.show_error(exc)

    def run_recognize(self) -> None:
        try:
            image_path = self.validate_image()
            board = self.validate_board()
            output_dir = Path(self.output_dir.get())
            already_rectified = bool(self.already_rectified.get())
            
            corners = None
            if not already_rectified:
                corners_str = self.corners.get().strip()
                if not corners_str:
                    raise ValueError("未勾選已校正時，必須提供四角座標")
                corners = parse_corners(corners_str)
                
            config = self.detector_config()
            allow_mirror = bool(self.allow_mirror.get())
            
            self.run_background_task(
                target=self._run_recognize_worker,
                args=(image_path, board, output_dir, already_rectified, corners, config, allow_mirror)
            )
        except Exception as exc:
            self.show_error(exc)

    def run_background_task(self, target, args) -> None:
        def worker() -> None:
            try:
                self.root.after(0, lambda: self.status.set("處理中..."))
                payload, preview_path = target(*args)
                self.root.after(0, lambda: self.show_result(payload, preview_path))
            except Exception as exc:
                self.root.after(0, lambda: self.show_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _run_label_worker(
        self,
        image_path: Path,
        output_dir: Path,
        config: DetectorConfig,
        save_crops: bool,
        preview_key: str
    ) -> tuple[dict, Path]:
        payload = label_pieces(
            image_path=image_path,
            output_dir=output_dir,
            detector_config=config,
            save_crops=save_crops,
            debug=True,
        )
        outputs = payload["outputs"]
        return payload, Path(outputs.get(preview_key) or outputs["original_numbered_path"])

    def _run_recognize_worker(
        self,
        image_path: Path,
        board: str,
        output_dir: Path,
        already_rectified: bool,
        corners: list[tuple[float, float]] | None,
        config: DetectorConfig,
        allow_mirror: bool
    ) -> tuple[dict, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = recognize(
            image_path=image_path,
            board_id=board,
            already_rectified=already_rectified,
            corners=corners,
            detector_config=config,
            matcher_config=MatcherConfig(allow_mirror=allow_mirror),
            debug=True,
        )

        debug = payload.get("debug", {})
        preview = Path(debug.get("match_overlay_path") or debug.get("rectified_image_path"))
        result_copy = output_dir / "recognize_result.json"
        with result_copy.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        payload["quick_validate_result_path"] = str(result_copy)
        return payload, preview

    def show_preview(self, image_path: str | Path) -> None:
        preview_path = make_preview_image(image_path)
        self.preview_image = tk.PhotoImage(file=str(preview_path))
        self.preview_label.configure(image=self.preview_image)

    def show_result(self, payload: dict, preview_path: str | Path) -> None:
        try:
            self.status.set("處理完成")
            self.show_preview(preview_path)
            self.current_payload = payload
            self.populate_results_table(payload)
        except Exception as exc:
            self.show_error(exc)

    def populate_results_table(self, payload: dict) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        pieces = payload.get("pieces", [])
        if not pieces:
            return

        for piece in pieces:
            piece_id = piece.get("piece_id", "unknown")
            
            if "matched_slot_id" in piece or "iou" in piece:
                matched_slot_id = piece.get("matched_slot_id") or "無 (None)"
                confidence = piece.get("confidence", 0.0)
                confidence_str = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else str(confidence)
                status = piece.get("status", "unknown")
                
                status_map = {
                    "confident": "確定 (Confident)",
                    "ambiguous": "模糊 (Ambiguous)",
                    "rejected": "拒絕 (Rejected)"
                }
                status_str = status_map.get(status, status)
                
                dx = piece.get("dx", 0)
                dy = piece.get("dy", 0)
                offset_str = f"({dx}, {dy})"
            else:
                matched_slot_id = "N/A (僅偵測)"
                confidence_str = "N/A"
                status_str = "N/A"
                offset_str = "N/A"

            self.tree.insert(
                "",
                tk.END,
                iid=piece_id,
                values=(piece_id, matched_slot_id, confidence_str, status_str, offset_str)
            )

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    def get_source_image_for_cropping(self) -> Path:
        if self.current_payload and "debug" in self.current_payload:
            rectified_path = self.current_payload["debug"].get("rectified_image_path")
            if rectified_path and Path(rectified_path).exists():
                return Path(rectified_path)
        return Path(self.image_path.get())

    def make_piece_crop_on_the_fly(self, bbox: list[int], max_width: int = 240, max_height: int = 240) -> Path | None:
        try:
            source_img_path = self.get_source_image_for_cropping()
            if not source_img_path or not source_img_path.exists():
                return None
            
            image = cv2.imread(str(source_img_path), cv2.IMREAD_COLOR)
            if image is None:
                return None
                
            x, y, w, h = bbox
            pad = 20
            x0 = max(0, int(x - pad))
            y0 = max(0, int(y - pad))
            x1 = min(image.shape[1], int(x + w + pad))
            y1 = min(image.shape[0], int(y + h + pad))
            
            cropped = image[y0:y1, x0:x1]
            if cropped.size == 0:
                return None
                
            cropped = cv2.copyMakeBorder(cropped, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[240, 240, 240])
            
            height, width = cropped.shape[:2]
            scale = min(max_width / width, max_height / height, 1.0)
            if scale < 1.0:
                cropped = cv2.resize(cropped, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                
            preview_dir = OUTPUTS_DIR / "gui_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / "piece_crop_preview.png"
            cv2.imwrite(str(preview_path), cropped)
            return preview_path
        except Exception as e:
            print(f"Error cropping piece: {e}")
            return None

    def on_tree_select(self, event) -> None:
        selected_items = self.tree.selection()
        if not selected_items:
            return

        piece_id = selected_items[0]
        if not self.current_payload:
            return

        pieces = self.current_payload.get("pieces", [])
        piece_dict = next((p for p in pieces if p.get("piece_id") == piece_id), None)
        if not piece_dict:
            return

        # 1. Update details JSON read-only textbox
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, json.dumps(piece_dict, ensure_ascii=False, indent=2))
        self.detail_text.config(state="disabled")

        # 2. Crop piece photo on the fly using bbox
        bbox = piece_dict.get("bbox")
        if bbox is None and "debug" in piece_dict:
            bbox = piece_dict["debug"].get("detected_piece", {}).get("bbox")

        if bbox and len(bbox) == 4:
            piece_crop_path = self.make_piece_crop_on_the_fly(bbox, max_width=240, max_height=240)
            if piece_crop_path and piece_crop_path.exists():
                try:
                    self.piece_crop_image = tk.PhotoImage(file=str(piece_crop_path))
                    self.piece_preview_label.configure(image=self.piece_crop_image, text="")
                except Exception as e:
                    self.piece_preview_label.configure(image="", text=f"載入失敗: {e}")
            else:
                self.piece_preview_label.configure(image="", text="影像切片失敗")
        else:
            self.piece_preview_label.configure(image="", text="無 BBox 資料")

        # 3. Update slot preview image if matched
        board_id = self.current_payload.get("board_id")
        matched_slot_id = piece_dict.get("matched_slot_id")

        if board_id and matched_slot_id:
            try:
                from puzzle_recognition.board_builder import load_board_config
                board_config = load_board_config(board_id)
                slot = next((s for s in board_config.get("slots", []) if s.get("slot_id") == matched_slot_id), None)
                
                if slot and "mask_path" in slot:
                    mask_relative_path = slot["mask_path"]
                    slot_mask_abs_path = BOARDS_DIR / board_id / mask_relative_path
                    
                    if slot_mask_abs_path.exists():
                        self.selected_slot_path = slot_mask_abs_path
                        preview_img_path = make_slot_preview_image(slot_mask_abs_path, max_width=240, max_height=240)
                        
                        self.slot_preview_image = tk.PhotoImage(file=str(preview_img_path))
                        self.slot_preview_label.configure(image=self.slot_preview_image, text="")
                        self.popup_slot_btn.config(state="normal")
                        return
            except Exception as e:
                self.slot_preview_label.configure(image="", text=f"載入預覽失敗: {str(e)}")
                self.popup_slot_btn.config(state="disabled")
                self.selected_slot_path = None
                return

        # Default fallback if no matched slot
        self.slot_preview_label.configure(image="", text="此拼圖無匹配底板孔位")
        self.popup_slot_btn.config(state="disabled")
        self.selected_slot_path = None

    def popup_slot_btn_clicked(self) -> None:
        if self.selected_slot_path and self.selected_slot_path.exists():
            self.popup_slot_mask(self.selected_slot_path)

    def popup_slot_mask(self, mask_path: Path | None = None) -> None:
        if mask_path is None:
            if not hasattr(self, "selected_slot_path") or not self.selected_slot_path:
                return
            mask_path = self.selected_slot_path

        if not mask_path.exists():
            messagebox.showerror("錯誤", f"底板遮罩檔案不存在：{mask_path}")
            return

        top = tk.Toplevel(self.root)
        top.title(f"完整底板遮罩 - {mask_path.name}")
        
        try:
            image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"無法讀取影像: {mask_path}")
            
            height, width = image.shape[:2]
            max_w, max_h = 800, 600
            scale = min(max_w / width, max_h / height, 1.0)
            if scale < 1.0:
                display_img = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            else:
                display_img = image

            preview_dir = OUTPUTS_DIR / "gui_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            temp_path = preview_dir / "temp_popup.png"
            cv2.imwrite(str(temp_path), display_img)

            popup_img = tk.PhotoImage(file=str(temp_path))
            top.popup_img = popup_img 
            
            lbl = ttk.Label(top, image=popup_img)
            lbl.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            info_lbl = ttk.Label(top, text=f"孔位路徑: {mask_path.relative_to(PROJECT_ROOT)}", font=("TkDefaultFont", 9, "italic"))
            info_lbl.pack(pady=(0, 10))
            
            top.focus_set()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法彈出完整遮罩窗口: {str(e)}")

    def show_error(self, exc: Exception) -> None:
        self.status.set("發生錯誤")
        messagebox.showerror("處理失敗", str(exc))


def main() -> None:
    root = tk.Tk()
    QuickValidateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
