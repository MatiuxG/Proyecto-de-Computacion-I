import customtkinter as ctk
import threading
import os
import pandas as pd
from datetime import datetime
from tkinter import messagebox, filedialog
import joblib

from modelos_entrenamiento.random_forest import entrenamiento_random_forest
from modelos_entrenamiento.decisiontree import entrenamiento_arbol_de_decision
from modelos_entrenamiento.svm import entrenamiento_svm
from modelos_entrenamiento.prediccion import realizar_prediccion

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neural Studio Pro - Madrid Mobility")
        self.geometry("1200x900")
        self.ruta_directorio_entrenamiento = os.getcwd()
        self.modelo_seleccionado_path = ""
        self.resultados_lote_df = pd.DataFrame()
        self.distritos = {
            1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 5: "Chamartín",
            6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 9: "Moncloa-Aravaca",
            10: "Latina", 11: "Carabanchel", 12: "Usera", 13: "Puente de Vallecas",
            14: "Moratalaz", 15: "Ciudad Lineal", 16: "Hortaleza", 17: "Villaverde",
            18: "Villa de Vallecas", 19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas"
        }
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        self.tab_train = self.tabview.add("1. Entrenamiento")
        self.tab_predict = self.tabview.add("2. Predicción")
        self.setup_train_ui()
        self.setup_predict_ui()

    #pantalla de entrenamiento
    def setup_train_ui(self):
        f_left = ctk.CTkFrame(self.tab_train)
        f_left.pack(side="left", padx=10, pady=10, fill="both")
        ctk.CTkLabel(f_left, text="CONFIGURACIÓN DEL MODELO", font=("Arial", 16, "bold")).pack(pady=10)
        ctk.CTkLabel(f_left, text="Objetivo a Predecir:").pack()
        self.target_menu = ctk.CTkOptionMenu(f_left, values=["Accidentes", "Calidad Aire", "Emergencias"], command=self.validar_anti_leakage)
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
        self.validar_anti_leakage(self.target_menu.get())
        ctk.CTkLabel(f_left, text="Algoritmo:").pack(pady=10)
        self.algo_menu = ctk.CTkOptionMenu(f_left, values=["Random Forest", "Decision Tree", "SVM"])
        self.algo_menu.pack(pady=5)
        self.ent_nombre_modelo = ctk.CTkEntry(f_left, placeholder_text="Nombre del modelo...")
        self.ent_nombre_modelo.pack(pady=10, fill="x", padx=10)
        ctk.CTkButton(f_left, text="📁 Elegir Carpeta", command=self.elegir_carpeta_entrenamiento).pack(pady=5)
        self.lbl_ruta_entreno = ctk.CTkLabel(f_left, text=f"Ruta: {self.ruta_directorio_entrenamiento}", font=("Arial", 10), wraplength=200)
        self.lbl_ruta_entreno.pack()
        self.btn_train = ctk.CTkButton(f_left, text="EJECUTAR ENTRENAMIENTO", fg_color="#2ecc71", text_color="black", command=self.run_train)
        self.btn_train.pack(pady=20, fill="x", padx=10)
        f_right = ctk.CTkFrame(self.tab_train, fg_color="#1a1a1a")
        f_right.pack(side="right", padx=10, pady=10, fill="both", expand=True)
        ctk.CTkLabel(f_right, text="FICHA TÉCNICA", text_color="white", font=("Arial", 16, "bold")).pack(pady=10)
        self.info_box = ctk.CTkTextbox(f_right, font=("Courier", 14), fg_color="#1a1a1a", text_color="#00ff00")
        self.info_box.pack(padx=10, pady=10, fill="both", expand=True)
        self.info_box.insert("0.0", ">>> Sistema listo.")

    def validar_anti_leakage(self, choice):
        mapping = {"Accidentes": "total_de_accidentes", "Calidad Aire": "valor_calidad_aire", "Emergencias": "cantidad_emergencias"}
        target_col = mapping[choice]
        for col, cb in self.check_boxes.items():
            if col == target_col:
                self.c_vars[col].set(False)
                cb.configure(state="disabled", text=cb.cget("text").split(" (")[0] + " (BLOQUEADO)")
            else:
                cb.configure(state="normal", text=col.replace("_"," ").title())

    def elegir_carpeta_entrenamiento(self):
        p = filedialog.askdirectory()
        if p: self.ruta_directorio_entrenamiento = p; self.lbl_ruta_entreno.configure(text=f"Ruta: {p}")

    def run_train(self):
        nombre = self.ent_nombre_modelo.get().strip()
        if nombre:
            ruta_final = os.path.join(self.ruta_directorio_entrenamiento, f"{nombre}.pkl")
            features = ["dia", "mes", "año", "codigo_de_distrito"]
            for k, v in self.c_vars.items():
                if v.get(): features.append(k)

            #asocia cada opcion del menu con su funcion de entrenamiento
            algoritmos = {
                "Random Forest": entrenamiento_random_forest,
                "Decision Tree": entrenamiento_arbol_de_decision,
                "SVM": entrenamiento_svm,
            }
            algo = self.algo_menu.get()
            entrenar = algoritmos.get(algo, entrenamiento_random_forest)

            def task():
                self.btn_train.configure(state="disabled", text="⌛ ENTRENANDO...")
                res = entrenar(self.target_menu.get(), features, ruta_final)

                self.info_box.delete("0.0", "end")
                self.info_box.insert("end", f"✅ MODELO GENERADO\n------------------\nAlgoritmo: {algo}\nFecha: {res['fecha']}\nAcc: {res['accuracy']:.2%}\nMAE: {res['mae']:.4f}\nMSE: {res['mse']:.4f}\nN: {res['n_muestras']}")

                self.btn_train.configure(state="normal", text="EJECUTAR ENTRENAMIENTO")
                messagebox.showinfo("Neural Studio", f"¡Entrenamiento finalizado!\nModelo guardado en: {nombre}.pkl")

            threading.Thread(target=task, daemon=True).start()
        else:
            messagebox.showwarning("Error", "Indique un nombre.")

    #pantalla de prediccion
    def setup_predict_ui(self):
        f_top = ctk.CTkFrame(self.tab_predict)
        f_top.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(f_top, text="Ejemplares (CSV):").grid(row=0, column=0, padx=10)
        self.ent_csv_ejemplares = ctk.CTkEntry(f_top, width=400)
        self.ent_csv_ejemplares.grid(row=0, column=1, padx=10)
        ctk.CTkButton(f_top, text="Abrir", command=self.cargar_csv_ejemplares).grid(row=0, column=2, padx=10)

        ctk.CTkLabel(f_top, text="Modelo (.pkl):").grid(row=1, column=0, padx=10)
        self.ent_model_path = ctk.CTkEntry(f_top, width=400)
        self.ent_model_path.grid(row=1, column=1, padx=10)
        ctk.CTkButton(f_top, text="Abrir", command=self.seleccionar_modelo_manual).grid(row=1, column=2, padx=10)

        f_mid = ctk.CTkFrame(self.tab_predict)
        f_mid.pack(padx=20, pady=10, fill="both", expand=True)

        self.txt_tabla_resultados = ctk.CTkTextbox(f_mid, font=("Courier", 12))
        self.txt_tabla_resultados.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.f_resumen = ctk.CTkFrame(f_mid, width=250)
        self.f_resumen.pack(side="right", padx=10, pady=10, fill="y")
        ctk.CTkLabel(self.f_resumen, text="RESUMEN:", font=("Arial", 14, "bold")).pack(pady=10)
        self.lbl_resumen_stats = ctk.CTkLabel(self.f_resumen, text="Total: 0\nAlto Riesgo: 0\n\nTiempo: 0s", justify="left")
        self.lbl_resumen_stats.pack(pady=20, padx=10)

        f_bot = ctk.CTkFrame(self.tab_predict)
        f_bot.pack(padx=20, pady=10, fill="x")
        self.btn_ejecutar = ctk.CTkButton(f_bot, text="EJECUTAR ANÁLISIS", fg_color="#e67e22", height=40, command=self.ejecutar_prediccion_lote)
        self.btn_ejecutar.pack(side="left", padx=20)
        ctk.CTkButton(f_bot, text="📥 EXPORTAR EXCEL", fg_color="#3498db", command=self.exportar_analisis_detallado).pack(side="right", padx=20)

    def cargar_csv_ejemplares(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if ruta: self.ent_csv_ejemplares.delete(0, "end"); self.ent_csv_ejemplares.insert(0, ruta)

    def seleccionar_modelo_manual(self):
        ruta = filedialog.askopenfilename(filetypes=[("Modelo", "*.pkl")])
        if ruta: self.ent_model_path.delete(0, "end"); self.ent_model_path.insert(0, ruta)

    def ejecutar_prediccion_lote(self):
        csv_p = self.ent_csv_ejemplares.get()
        mod_p = self.ent_model_path.get()
        if csv_p and mod_p:
            def thread_task():
                self.btn_ejecutar.configure(state="disabled", text="⌛ PENSANDO...")
                try:
                    df = pd.read_csv(csv_p, sep=None, engine="python")
                    start = datetime.now()
                    preds = []

                    self.txt_tabla_resultados.delete("1.0", "end")
                    self.txt_tabla_resultados.insert("end", f"{'Fila':<5} | {'Fecha':<10} | {'Distrito':<15} | {'Prob':<6}\n" + "-"*50)

                    for idx, row in df.iterrows():
                        res = realizar_prediccion(mod_p, row.to_dict())
                        prob = res.get("probabilidad", 0) if isinstance(res, dict) else 0
                        preds.append(prob)

                        if idx < 100:
                            dia = int(row.get('Dia', row.get('dia', 1)))
                            mes = int(row.get('Mes', row.get('mes', 1)))
                            dist_id = int(row.get('district_code', row.get('codigo_de_distrito', 0)))
                            nombre_dist = self.distritos.get(dist_id, f"ID {dist_id}")
                            self.txt_tabla_resultados.insert("end", f"\n{idx+1:<5} | {dia:02d}/{mes:02d} | {nombre_dist:<15} | {prob:.1%}")

                    self.resultados_lote_df = df.copy()
                    self.resultados_lote_df["Probabilidad_Riesgo"] = preds
                    self.resultados_lote_df["Nivel_Alerta"] = ["ALTO" if p > 0.6 else "MEDIO" if p > 0.3 else "BAJO" for p in preds]

                    t = (datetime.now() - start).total_seconds()
                    alto = len([p for p in preds if p > 0.6])
                    self.lbl_resumen_stats.configure(text=f"Total: {len(df)}\nAlto Riesgo: {alto}\nTiempo: {t:.2f}s")

                    #avisa de que el analisis termino
                    messagebox.showinfo("Análisis Completado", f"Se han procesado {len(df)} ejemplares con éxito.")

                except Exception as e:
                    messagebox.showerror("Error", f"Error procesando el análisis: {e}")
                finally:
                    self.btn_ejecutar.configure(state="normal", text="EJECUTAR ANÁLISIS")

            threading.Thread(target=thread_task, daemon=True).start()
        else:
            messagebox.showwarning("Atención", "Seleccione primero el CSV y el Modelo.")

    def exportar_analisis_detallado(self):
        if self.resultados_lote_df.empty:
            messagebox.showwarning("Aviso", "No hay resultados. Primero ejecute el análisis.")
        else:
            p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
            if p:
                try:
                    self.resultados_lote_df.to_excel(p, index=False)
                    messagebox.showinfo("Exportación", "El archivo Excel se ha generado correctamente.")
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
