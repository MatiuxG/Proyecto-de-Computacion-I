import customtkinter as ctk
import threading
import os
import pandas as pd
from datetime import datetime
from tkinter import messagebox, filedialog
import joblib

# Importación de la lógica de entrenamiento y predicción
from modelos_entrenamiento.random_forest import entrenamiento_random_forest
from modelos_entrenamiento.decisiontree import entrenamiento_arbol_de_decision
from modelos_entrenamiento.svm import entrenamiento_svm
from modelos_entrenamiento.prediccion import realizar_prediccion

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neural Studio Pro - Madrid Mobility")
        self.geometry("1100x850")
        
        self.ruta_directorio = os.getcwd()
        self.modelo_seleccionado_path = "" 
        self.historial = [] # Se usará para el Excel detallado
        self.distritos = {
            1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín", 
            6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca", 
            10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas", 
            14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde", 
            18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas"
        }

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.t1 = self.tabview.add("1. Entrenamiento")
        self.t2 = self.tabview.add("2. Predicción")

        self.setup_t1()
        self.setup_t2()

    def setup_t1(self):
        f_left = ctk.CTkFrame(self.t1)
        f_left.pack(side="left", padx=10, pady=10, fill="both")

        ctk.CTkLabel(f_left, text="CONFIGURACIÓN DEL MODELO", font=("Arial", 16, "bold")).pack(pady=10)

        ctk.CTkLabel(f_left, text="Objetivo a Predecir:").pack()
        self.target_menu = ctk.CTkOptionMenu(f_left, 
                                            values=["Accidentes", "Calidad Aire", "Emergencias"],
                                            command=self.validar_sources)
        self.target_menu.pack(pady=5)

        ctk.CTkLabel(f_left, text="Fuentes de Datos (Features):").pack(pady=10)
        self.check_boxes = {}
        self.c_vars = {
            "trafico_medio": ctk.BooleanVar(value=True),
            "valor_calidad_aire": ctk.BooleanVar(value=True),
            "cantidad_emergencias": ctk.BooleanVar(value=True),
            "total_de_accidentes": ctk.BooleanVar(value=True)
        }
        
        for k, v in self.c_vars.items():
            cb = ctk.CTkCheckBox(f_left, text=k.replace("_"," ").title(), variable=v)
            cb.pack(anchor="w", padx=20, pady=2)
            self.check_boxes[k] = cb

        self.validar_sources(self.target_menu.get())

        ctk.CTkLabel(f_left, text="Algoritmo:").pack(pady=10)
        self.algo_menu = ctk.CTkOptionMenu(f_left, values=["Random Forest", "Decision Tree", "SVM"])
        self.algo_menu.pack(pady=5)

        ctk.CTkLabel(f_left, text="Nombre del Modelo:").pack(pady=10)
        self.ent_nombre = ctk.CTkEntry(f_left, placeholder_text="ej: modelo_v1")
        self.ent_nombre.pack(pady=5, fill="x", padx=10)

        ctk.CTkButton(f_left, text="📁 Elegir Carpeta de Guardado", command=self.choose_dir).pack(pady=10)
        self.lbl_path = ctk.CTkLabel(f_left, text=f"Ruta: {self.ruta_directorio}", font=("Arial", 10), wraplength=200)
        self.lbl_path.pack()

        self.btn_train = ctk.CTkButton(f_left, text="🚀 ENTRENAR Y GUARDAR", fg_color="#2ecc71", text_color="black", font=("Arial", 12, "bold"), command=self.run_train)
        self.btn_train.pack(pady=20, fill="x", padx=10)

        f_right = ctk.CTkFrame(self.t1, fg_color="#1a1a1a")
        f_right.pack(side="right", padx=10, pady=10, fill="both", expand=True)
        ctk.CTkLabel(f_right, text="📊 FICHA TÉCNICA", text_color="white", font=("Arial", 16, "bold")).pack(pady=10)
        self.info_box = ctk.CTkTextbox(f_right, font=("Courier", 14), fg_color="#1a1a1a", text_color="#00ff00")
        self.info_box.pack(padx=10, pady=10, fill="both", expand=True)
        self.info_box.insert("0.0", ">>> Esperando entrenamiento...")

    def validar_sources(self, choice):
        mapping = {"Accidentes": "total_de_accidentes", "Calidad Aire": "valor_calidad_aire", "Emergencias": "cantidad_emergencias"}
        target_col = mapping[choice]
        for col, cb in self.check_boxes.items():
            if col == target_col:
                self.c_vars[col].set(False)
                cb.configure(state="disabled", text=cb.cget("text") + " (BLOQUEADO)")
            else:
                cb.configure(state="normal", text=col.replace("_"," ").title())

    def setup_t2(self):
        f_in = ctk.CTkFrame(self.t2)
        f_in.pack(pady=20, fill="x", padx=20)
        
        self.entry_f = ctk.CTkEntry(f_in, placeholder_text="DD/MM/AAAA")
        self.entry_f.grid(row=0, column=0, padx=10)
        
        self.dist_combo = ctk.CTkOptionMenu(f_in, values=[f"{k}-{v}" for k,v in self.distritos.items()])
        self.dist_combo.grid(row=0, column=1, padx=10)
        
        ctk.CTkButton(f_in, text="📂 Seleccionar Modelo (.pkl)", command=self.buscar_modelo_manual).grid(row=0, column=2, padx=10)
        
        self.lbl_modelo_status = ctk.CTkLabel(self.t2, text="Ningún modelo seleccionado", font=("Arial", 11, "italic"))
        self.lbl_modelo_status.pack(pady=5)

        ctk.CTkButton(self.t2, text="🔍 CALCULAR PREDICCIÓN", command=self.do_pred).pack(pady=20)
        self.res_lbl = ctk.CTkLabel(self.t2, text="0.0%", font=("Arial", 60, "bold"))
        self.res_lbl.pack(pady=20)

        ctk.CTkButton(self.t2, text="📥 EXPORTAR ANÁLISIS DETALLADO", fg_color="#3498db", command=self.export).pack(pady=10)

    def buscar_modelo_manual(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de modelo", "*.pkl")])
        if archivo:
            self.modelo_seleccionado_path = archivo
            self.lbl_modelo_status.configure(text=f"Modelo cargado: {os.path.basename(archivo)}", text_color="#2ecc71")

    def choose_dir(self):
        p = filedialog.askdirectory()
        if p: self.ruta_directorio = p; self.lbl_path.configure(text=f"Ruta: {p}")

    def run_train(self):
        nombre = self.ent_nombre.get().strip()
        if not nombre: messagebox.showwarning("Faltan datos", "Indique nombre del modelo."); return
        
        ruta_final = os.path.join(self.ruta_directorio, f"{nombre}.pkl")
        target = self.target_menu.get()
        algo = self.algo_menu.get()
        features = ["dia", "mes", "año", "codigo_de_distrito"]
        for k, v in self.c_vars.items():
            if v.get(): features.append(k)

        def task():
            self.info_box.delete("0.0", "end")
            if algo == "Random Forest": res = entrenamiento_random_forest(target, features, ruta_final)
            elif algo == "Decision Tree": res = entrenamiento_arbol_de_decision(target, features, ruta_final)
            else: res = entrenamiento_svm(target, features, ruta_final)

            if "error" in res: self.info_box.insert("end", f"\n[!] ERROR: {res['error']}")
            else:
                txt = (f"✅ MODELO GUARDADO\n--------------------------\n🎯 Accuracy: {res['accuracy']:.2%}\n"
                       f"📉 MAE: {res['mae']:.4f}\n📉 MSE: {res['mse']:.4f}\n🔢 Muestras: {res['n_muestras']}")
                self.info_box.insert("end", txt)
        threading.Thread(target=task).start()

    def do_pred(self):
        if not self.modelo_seleccionado_path: messagebox.showwarning("Atención", "Cargue un modelo."); return
        try:
            f_str = self.entry_f.get()
            f = datetime.strptime(f_str, "%d/%m/%Y")
            d_val = self.dist_combo.get()
            d_id = int(d_val.split("-")[0])
            d_nom = d_val.split("-")[1]
            
            # Datos de entrada
            datos = {'dia': f.day, 'mes': f.month, 'año': f.year, 'codigo_de_distrito': d_id}
            
            res = realizar_prediccion(self.modelo_seleccionado_path, datos)
            if "error" in res: messagebox.showerror("Error", res["error"])
            else:
                p = res['probabilidad']
                self.res_lbl.configure(text=f"{p:.1%}", text_color=("red" if p > 0.5 else "green"))
                
                # --- EXPORTACIÓN DETALLADA: Guardamos el contexto completo ---
                self.historial.append({
                    "Timestamp_Analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Modelo_Utilizado": os.path.basename(self.modelo_seleccionado_path),
                    "Fecha_Prediccion": f_str,
                    "ID_Distrito": d_id,
                    "Nombre_Distrito": d_nom,
                    "Probabilidad_Riesgo": f"{p:.2%}",
                    "Nivel_Alerta": "ALTO" if p > 0.6 else "MEDIO" if p > 0.3 else "BAJO"
                })
        except Exception as e: messagebox.showerror("Error", f"Fallo: {e}")

    def export(self):
        if not self.historial: messagebox.showwarning("Vacío", "Sin datos para exportar."); return
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            df = pd.DataFrame(self.historial)
            # Ordenar columnas para que el Excel sea legible
            cols = ["Timestamp_Analisis", "Fecha_Prediccion", "Nombre_Distrito", "Probabilidad_Riesgo", "Nivel_Alerta", "Modelo_Utilizado"]
            df[cols].to_excel(p, index=False)
            messagebox.showinfo("Éxito", "Análisis detallado exportado.")

if __name__ == "__main__":
    app = App(); app.mainloop()