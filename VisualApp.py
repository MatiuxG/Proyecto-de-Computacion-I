import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
# Asegurar backend correcto para Tkinter
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import random
import os

# 🎨 --- Configuración Inicial ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Colores y fuentes
TEXT_COLOR_DARK = "gray10"
ACCENT_COLOR = "#3B8ED0"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración Ventana
        self.title("AI Model Trainer - Studio")
        self.geometry("1200x800")

        # Variables de Estado para Predicción
        self.pred_file_path = None
        self.model_file_path = None
        self.prediction_results = [] # Almacenará los resultados generados

        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Crear Sidebar
        self.create_sidebar()

        # 2. Inicializar Frames
        self.home_frame = None 
        self.results_frame = None
        self.prediction_config_frame = None # Pantalla 3
        self.prediction_results_frame = None # Pantalla 4

        self.create_home_frame()            # Pantalla Configuración/Entrenamiento
        self.create_results_frame()         # Pantalla Resultados Entrenamiento
        self.create_prediction_config_frame() # Pantalla Configuración Predicción
        self.create_prediction_results_frame() # Pantalla Resultados Predicción

        # 3. Iniciar en Home
        self.select_frame("home")

    # ============================================================
    # SIDEBAR
    # ============================================================
    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NEURAL\nSTUDIO", 
                                     font=ctk.CTkFont(size=20, weight="bold"), 
                                     text_color=TEXT_COLOR_DARK) 
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- Botones de Navegación ---
        # 1. Entrenamiento (Home)
        self.btn_nav_train = ctk.CTkButton(self.sidebar_frame, text="Entrenamiento", fg_color="white",border_width=2, text_color=TEXT_COLOR_DARK,
                                          command=lambda: self.select_frame("home"))
        self.btn_nav_train.grid(row=1, column=0, padx=20, pady=10)
        
        # 2. Predicción (NUEVO)
        self.btn_nav_predict = ctk.CTkButton(self.sidebar_frame, text="Predicción", fg_color="white", 
                                            border_width=2, text_color=TEXT_COLOR_DARK,
                                            command=lambda: self.select_frame("prediction_config"))
        self.btn_nav_predict.grid(row=2, column=0, padx=20, pady=10)

        # 3. Historial/Config
        self.btn_nav_history = ctk.CTkButton(self.sidebar_frame, text="Historial", fg_color="white", 
                                            border_width=2, text_color=TEXT_COLOR_DARK, 
                                            command=lambda: self.select_frame("results"))
        self.btn_nav_history.grid(row=3, column=0, padx=20, pady=10)

        # 4. Botón SALIR
        self.btn_exit = ctk.CTkButton(self.sidebar_frame, text="Salir", 
                                     fg_color="#C70039", hover_color="#A00030",
                                     command=self.exit_program)
        self.btn_exit.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="s")

    def exit_program(self):
        if messagebox.askyesno("Salir", "¿Estás seguro de que deseas salir del programa?"):
            self.quit()

    def select_frame(self, name):
        # Resetear colores botones
        self.btn_nav_train.configure(fg_color="white")
        self.btn_nav_predict.configure(fg_color="white")
        self.btn_nav_history.configure(fg_color="white")

        # Ocultar todos
        if self.home_frame: self.home_frame.grid_forget()
        if self.results_frame: self.results_frame.grid_forget()
        if self.prediction_config_frame: self.prediction_config_frame.grid_forget()
        if self.prediction_results_frame: self.prediction_results_frame.grid_forget()

        # Mostrar seleccionado y activar color botón
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_train.configure(fg_color=("gray75", "gray25"))
        elif name == "results":
            self.results_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_history.configure(fg_color=("gray75", "gray25"))
        elif name == "prediction_config":
            self.prediction_config_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_predict.configure(fg_color=("gray75", "gray25"))
        elif name == "prediction_results":
            self.prediction_results_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
            self.btn_nav_predict.configure(fg_color=("gray75", "gray25"))

    # ============================================================
    # PANTALLA 1: CONFIGURACIÓN DEL ENTRENAMIENTO
    # ============================================================
    def create_home_frame(self):
        self.home_frame = ctk.CTkFrame(self, fg_color="white")
        self.home_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.home_frame, text="Configuración del Entrenamiento", 
                     font=ctk.CTkFont(size=26, weight="bold"), 
                     text_color=TEXT_COLOR_DARK).pack(anchor="w", pady=(0,20))

        upload_frame = ctk.CTkFrame(self.home_frame, fg_color="white", border_width=2, border_color="gray70")
        upload_frame.pack(fill="x", pady=10, ipady=20)
        
        ctk.CTkLabel(upload_frame, text="Drag & Drop CSV/Excel here", text_color="gray50").pack()
        ctk.CTkButton(upload_frame, text="Explorar Archivos", command=self.browse_file_train, 
                      fg_color="gray30", width=120).pack(pady=5)
        self.lbl_filename_train = ctk.CTkLabel(upload_frame, text="", text_color=ACCENT_COLOR, font=("Arial", 12, "bold"))
        self.lbl_filename_train.pack()

        settings_frame = ctk.CTkFrame(self.home_frame, fg_color="white")
        settings_frame.pack(fill="x", pady=20)

        ctk.CTkLabel(settings_frame, text="Objetivo de Entrenamiento:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        self.combo_target = ctk.CTkOptionMenu(settings_frame, values=["Emergencias", "Tráfico", "Accidentes", "Todos (Multi-clase)"])
        self.combo_target.pack(fill="x", pady=(5, 15))

        ctk.CTkLabel(settings_frame, text="Algoritmo de ML:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10,0))
        self.combo_algorithm = ctk.CTkOptionMenu(settings_frame, 
                                                 values=["Random Forest", "Decision Tree", "SVM (Support Vector Machine)"])
        self.combo_algorithm.pack(fill="x", pady=(5, 15))

        ctk.CTkLabel(settings_frame, text="Configuración del Modelo:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        self.radio_var = tk.IntVar(value=0)
        self.radio_1 = ctk.CTkRadioButton(settings_frame, text="Entrenamiento Individual (Rápido)", variable=self.radio_var, value=0)
        self.radio_1.pack(anchor="w", pady=5)
        self.radio_2 = ctk.CTkRadioButton(settings_frame, text="Entrenamiento Simultáneo (3 Modelos - Lento)", variable=self.radio_var, value=1)
        self.radio_2.pack(anchor="w", pady=5)

        ctk.CTkFrame(self.home_frame, height=20, fg_color="white").pack()

        self.btn_start = ctk.CTkButton(self.home_frame, text="EJECUTAR ENTRENAMIENTO", height=50, 
                                      fg_color="#2CC985", hover_color="#24A36B", 
                                      font=ctk.CTkFont(size=16, weight="bold"),
                                      command=self.start_training_thread)
        self.btn_start.pack(fill="x", pady=10)
        
        self.lbl_status = ctk.CTkLabel(self.home_frame, text="", text_color="gray40")
        self.lbl_status.pack()
        
        self.progress = ctk.CTkProgressBar(self.home_frame)
        self.progress.set(0)
        self.progress.pack_forget() 

    def browse_file_train(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", ".csv"), ("Excel Files", ".xlsx")])
        if filename:
            name_only = os.path.basename(filename)
            self.lbl_filename_train.configure(text=f"📄 {name_only} (Cargado)")

    def start_training_thread(self):
        self.btn_start.configure(state="disabled")
        self.progress.pack(fill="x", pady=10)
        threading.Thread(target=self.simulate_training_logic).start()

    def simulate_training_logic(self):
        algorithm = self.combo_algorithm.get()
        steps = ["Preprocesando datos...", f"Entrenando {algorithm}...", "Validando resultados...", "Finalizando..."]
        for i, step in enumerate(steps):
            self.lbl_status.configure(text=f"{step} {25*(i+1)}%")
            self.progress.set((i+1)/len(steps))
            time.sleep(0.8)
        self.after(500, self.training_complete)

    def training_complete(self):
        self.lbl_status.configure(text="")
        self.progress.pack_forget()
        self.btn_start.configure(state="normal")
        self.select_frame("results")

    # ============================================================
    # PANTALLA 2: RESULTADOS ENTRENAMIENTO
    # ============================================================
    def create_results_frame(self):
        self.results_frame = ctk.CTkFrame(self, fg_color="white")
        
        header = ctk.CTkFrame(self.results_frame, fg_color="white")
        header.pack(fill="x")
        ctk.CTkLabel(header, text="Resultados del Modelo", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR_DARK).pack(side="left")
        ctk.CTkLabel(header, text="ID: #TRN-2023-884", text_color="gray60").pack(side="right", pady=10)

        kpi_container = ctk.CTkFrame(self.results_frame, fg_color="white")
        kpi_container.pack(fill="x", pady=15)
        
        def create_kpi(parent, title, value, color):
            card = ctk.CTkFrame(parent, fg_color="white", border_width=1, border_color="gray85")
            card.pack(side="left", fill="x", expand=True, padx=5)
            ctk.CTkLabel(card, text=title, text_color="gray50", font=("Arial", 12)).pack(pady=(10,0))
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(pady=(0,10))

        create_kpi(kpi_container, "Tiempo Entr.", "00:15:30", "#333")
        create_kpi(kpi_container, "Precisión Global", "94.5%", "#2CC985")
        create_kpi(kpi_container, "Pérdida (Loss)", "0.021", "#C70039")

        graph_frame = ctk.CTkFrame(self.results_frame, fg_color="white")
        graph_frame.pack(fill="both", expand=True, pady=10)

        plt.style.use('default') 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.subplots_adjust(wspace=0.3, bottom=0.2)
        fig.patch.set_facecolor('#F3F3F3')

        epochs = range(1, 11)
        ax1.plot(epochs, [0.4,0.5,0.65,0.70,0.80,0.85,0.88,0.91,0.93,0.945], label='Accuracy', color='#2CC985', linewidth=2)
        ax1.set_title("Curva de Aprendizaje", fontsize=10)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_facecolor('white')

        conf_matrix = np.array([[45, 2, 3], [5, 38, 7], [1, 4, 45]]) 
        classes = ["Emerg.", "Tráfico", "Accid."]
        im = ax2.imshow(conf_matrix, cmap="Blues")
        ax2.set_title("Matriz de Confusión", fontsize=10)
        ax2.set_xticks(np.arange(len(classes)))
        ax2.set_yticks(np.arange(len(classes)))
        ax2.set_xticklabels(classes)
        ax2.set_yticklabels(classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax2.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        footer = ctk.CTkFrame(self.results_frame, height=60, fg_color="white")
        footer.pack(fill="x", pady=10)

        self.btn_save = ctk.CTkButton(footer, text="💾 Guardar Modelo (.pkl)", 
                                     fg_color="#3B8ED0", width=180, command=self.save_model)
        self.btn_save.pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(footer, text="Nombre del Reporte:", text_color="gray50").pack(side="left", padx=(20, 5))
        self.entry_report_name = ctk.CTkEntry(footer, width=200, placeholder_text="reporte_modelo")
        self.entry_report_name.pack(side="left", padx=5)

        self.opt_export = ctk.CTkOptionMenu(footer, values=["Excel (.xlsx)", "Texto (.txt)", "JSON"], width=130,
                                    fg_color="white", text_color="black", button_color="gray80")
        self.opt_export.pack(side="left", pady=10, padx=5)
        
        ctk.CTkButton(footer, text="Descargar", width=100, fg_color="gray30", 
                      command=self.export_report).pack(side="left", padx=5)

    def save_model(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", ".pkl")])
        if filename: messagebox.showinfo("Éxito", f"Modelo guardado en:\n{filename}")

    def export_report(self):
        messagebox.showinfo("Exportar", "Funcionalidad de exportación simulada correctamente.")

    # ============================================================
    # PANTALLA 3: CONFIGURACIÓN DE PREDICCIÓN (ACTUALIZADA)
    # ============================================================
    def create_prediction_config_frame(self):
        self.prediction_config_frame = ctk.CTkFrame(self, fg_color="white")
        
        # Título
        ctk.CTkLabel(self.prediction_config_frame, text="Configuración de Predicción", 
                     font=ctk.CTkFont(size=26, weight="bold"), 
                     text_color=TEXT_COLOR_DARK).pack(anchor="w", pady=(0,20))

        # --- Contenedor Principal dividido ---
        container = ctk.CTkFrame(self.prediction_config_frame, fg_color="white")
        container.pack(fill="both", expand=True, padx=10)

        # 1. Doble Área de Carga
        load_frame = ctk.CTkFrame(container, fg_color="gray95")
        load_frame.pack(fill="x", pady=10, padx=10)

        # 1.1 Cargar Datos Nuevos
        ctk.CTkLabel(load_frame, text="1. Fuente de Datos (Predecir)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        
        # Botón para navegar archivos reales
        self.btn_load_pred_data = ctk.CTkButton(load_frame, text="📂 Subir archivo a predecir (.csv/.xlsx)", 
                                              fg_color="gray70", command=self.browse_prediction_data)
        self.btn_load_pred_data.pack(fill="x", padx=10, pady=5)
        
        # Label para mostrar archivo seleccionado
        self.lbl_pred_file = ctk.CTkLabel(load_frame, text="Ningún archivo seleccionado", text_color="gray50", font=("Arial", 11))
        self.lbl_pred_file.pack(anchor="w", padx=15, pady=(0,10))

        # 1.2 Cargar Modelo
        ctk.CTkLabel(load_frame, text="2. Modelo Entrenado", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        
        model_selection_frame = ctk.CTkFrame(load_frame, fg_color="transparent")
        model_selection_frame.pack(fill="x", padx=10, pady=5)
        
        # Botón para cargar modelo real
        self.btn_load_model = ctk.CTkButton(model_selection_frame, text="📂 Seleccionar Modelo (.pkl)", 
                                            fg_color="gray70", command=self.browse_model_file)
        self.btn_load_model.pack(fill="x")

        # Label para mostrar modelo seleccionado
        self.lbl_model_file = ctk.CTkLabel(load_frame, text="Ningún modelo cargado", text_color="gray50", font=("Arial", 11))
        self.lbl_model_file.pack(anchor="w", padx=15, pady=(0,10))

        # 2. Botón de Acción
        self.btn_run_prediction = ctk.CTkButton(container, text="⚡ REALIZAR PREDICCIÓN", height=50,
                                               fg_color=ACCENT_COLOR, font=ctk.CTkFont(size=16, weight="bold"),
                                               command=self.run_prediction_process)
        self.btn_run_prediction.pack(fill="x", padx=10, pady=20)

        # 3. Log de Consola
        ctk.CTkLabel(container, text="Log de Ejecución:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10)
        self.console_log = ctk.CTkTextbox(container, height=200, fg_color="black", text_color="#00FF00", font=("Consolas", 12))
        self.console_log.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.console_log.insert("0.0", "Sistema listo. Esperando archivos de entrada...\n")
        self.console_log.configure(state="disabled")

    def browse_prediction_data(self):
        """Abre explorador para archivo de datos"""
        filename = filedialog.askopenfilename(filetypes=[("Archivos de Datos", "*.csv *.xlsx")])
        if filename:
            self.pred_file_path = filename
            name_only = os.path.basename(filename)
            self.lbl_pred_file.configure(text=f"✅ {name_only}", text_color="green")
            self.log_to_console(f"Archivo de datos cargado: {name_only}")
        else:
            self.log_to_console("Carga de datos cancelada.")

    def browse_model_file(self):
        """Abre explorador para modelo"""
        filename = filedialog.askopenfilename(filetypes=[("Modelos Pickle", "*.pkl"), ("Modelos H5", "*.h5")])
        if filename:
            self.model_file_path = filename
            name_only = os.path.basename(filename)
            self.lbl_model_file.configure(text=f"✅ {name_only}", text_color="green")
            self.log_to_console(f"Modelo cargado: {name_only}")
        else:
            self.log_to_console("Carga de modelo cancelada.")

    def log_to_console(self, message):
        self.console_log.configure(state="normal")
        self.console_log.insert("end", f"> {message}\n")
        self.console_log.see("end")
        self.console_log.configure(state="disabled")

    def run_prediction_process(self):
        # Validación: No correr si no hay archivos
        if not self.pred_file_path or not self.model_file_path:
            messagebox.showwarning("Faltan Archivos", "Por favor sube un archivo de datos y un modelo entrenado antes de continuar.")
            self.log_to_console("ERROR: Intento de predicción sin archivos.")
            return

        self.btn_run_prediction.configure(state="disabled")
        threading.Thread(target=self.simulate_prediction_logic).start()

    def simulate_prediction_logic(self):
        # Simulación basada en archivos reales (o placeholder si no se parsea)
        data_name = os.path.basename(self.pred_file_path)
        model_name = os.path.basename(self.model_file_path)

        actions = [
            f"Leyendo dataset: {data_name}...",
            f"Cargando modelo: {model_name}...",
            "Normalizando datos de entrada...",
            "Ejecutando inferencia...",
            "Generando coordenadas geográficas...",
            "Predicción finalizada con éxito."
        ]
        
        for action in actions:
            time.sleep(random.uniform(0.5, 1.2))
            self.after(0, lambda a=action: self.log_to_console(a))
        
        self.after(1000, self.finish_prediction)

    def finish_prediction(self):
        self.btn_run_prediction.configure(state="normal")
        self.log_to_console("Transfiriendo a vista de resultados...")
        
        # Generar datos "reales" solo ahora que terminó el proceso
        self.generate_prediction_results() 
        
        self.after(500, lambda: self.select_frame("prediction_results"))

    # ============================================================
    # PANTALLA 4: RESULTADOS DE PREDICCIÓN (ACTUALIZADA)
    # ============================================================
    def create_prediction_results_frame(self):
        self.prediction_results_frame = ctk.CTkFrame(self, fg_color="white")
        
        header = ctk.CTkFrame(self.prediction_results_frame, fg_color="white")
        header.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(header, text="Visualización Geográfica", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR_DARK).pack(side="left")
        
        self.tab_view = ctk.CTkTabview(self.prediction_results_frame)
        self.tab_view.pack(fill="both", expand=True)
        self.tab_view.add("Vista Mapa")
        self.tab_view.add("Vista Tabla")
        
        # --- TAB 1: MAPA ---
        self.map_frame = ctk.CTkFrame(self.tab_view.tab("Vista Mapa"), fg_color="white")
        self.map_frame.pack(fill="both", expand=True)
        
        self.map_fig, self.map_ax = plt.subplots(figsize=(6, 4))
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=self.map_frame)
        self.map_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Inicializar mapa VACÍO (sin puntos)
        self.map_ax.set_facecolor('#E6E6E6')
        self.map_ax.grid(True, color='white', linestyle='-', linewidth=2)
        self.map_ax.set_title("Esperando predicción...", fontsize=12)
        self.map_ax.set_xlabel("Longitud")
        self.map_ax.set_ylabel("Latitud")

        toolbar = NavigationToolbar2Tk(self.map_canvas, self.map_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        # --- TAB 2: TABLA ---
        self.table_frame = ctk.CTkFrame(self.tab_view.tab("Vista Tabla"), fg_color="transparent")
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Filtros
        filter_frame = ctk.CTkFrame(self.table_frame)
        filter_frame.pack(fill="x", pady=(0,10))
        
        # Checkboxes corregidos
        cb1 = ctk.CTkCheckBox(filter_frame, text="Emergencias (Rojo)")
        cb1.pack(side="left", padx=10)
        cb1.select()

        cb2 = ctk.CTkCheckBox(filter_frame, text="Tráfico (Amarillo)")
        cb2.pack(side="left", padx=10)
        cb2.select()

        cb3 = ctk.CTkCheckBox(filter_frame, text="Accidentes (Azul)")
        cb3.pack(side="left", padx=10)
        cb3.select()

        # Treeview VACÍO inicialmente
        columns = ("ID", "Lat", "Lon", "Clase", "Probabilidad")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        self.tree.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def generate_prediction_results(self):
        """Genera los puntos y actualiza mapa/tabla SOLO cuando se llama explícitamente"""
        self.map_ax.clear()
        
        # Simulamos que leemos el archivo y obtenemos N puntos
        n_points = 50 
        lats = np.random.uniform(40.3, 40.5, n_points)
        lons = np.random.uniform(-3.8, -3.6, n_points)
        categories = np.random.choice([0, 1, 2], n_points) 
        
        colors = ['#C70039', '#FFC300', '#3B8ED0'] 
        labels = ['Emergencia', 'Tráfico', 'Accidente']
        
        # Dibujar puntos en el mapa
        for i in range(3):
            mask = categories == i
            self.map_ax.scatter(lons[mask], lats[mask], c=colors[i], label=labels[i], s=100, alpha=0.7, edgecolors='white')
        
        self.map_ax.set_facecolor('#E6E6E6')
        self.map_ax.grid(True, color='white', linestyle='-', linewidth=2)
        self.map_ax.set_title("Resultados de Predicción", fontsize=12)
        self.map_ax.set_xlabel("Longitud")
        self.map_ax.set_ylabel("Latitud")
        self.map_ax.legend(loc='upper right')
        
        # Tooltip ejemplo
        random_idx = random.randint(0, n_points-1)
        self.map_ax.annotate(f"Pred: {labels[categories[random_idx]]}\nProb: {random.randint(80,99)}%", 
                             xy=(lons[random_idx], lats[random_idx]), 
                             xytext=(lons[random_idx]+0.01, lats[random_idx]+0.01),
                             arrowprops=dict(facecolor='black', shrink=0.05),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        self.map_canvas.draw()
        
        # Actualizar Tabla
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for i in range(n_points):
            self.tree.insert("", "end", values=(
                f"EVT-{1000+i}", 
                f"{lats[i]:.4f}", 
                f"{lons[i]:.4f}", 
                labels[categories[i]], 
                f"{random.randint(70, 99)}%"
            ))

if __name__ == "__main__":
    app = App()
    app.mainloop()