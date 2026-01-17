import customtkinter as ctk
import threading
from datetime import datetime
import joblib
import pandas as pd
import os
from tkinter import messagebox

# Importamos tus funciones de entrenamiento
from modelos_entrenamiento.random_forest import entrenamiento_random_forest
from modelos_entrenamiento.decisiontree import entrenamiento_arbol_de_decision
from modelos_entrenamiento.svm import entrenamiento_svm

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neural Studio - Predicción Madrid")
        self.geometry("850x700")

        self.distritos = {
         1: "Centro", 2: "Arganzuela", 3: "Retiro", 4: "Salamanca", 
            5: "Chamartín", 6: "Tetuán", 7: "Chamberí", 8: "Fuencarral-El Pardo", 
            9: "Moncloa-Aravaca", 10: "Latina", 11: "Carabanchel", 12: "Usera", 
            13: "Puente de Vallecas", 14: "Moratalaz", 15: "Ciudad Lineal", 
            16: "Hortaleza", 17: "Villaverde", 18: "Villa de Vallecas", 
            19: "Vicálvaro", 20: "San Blas-Canillejas", 21: "Barajas"
        }

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.tab_train = self.tabview.add("1. Entrenamiento")
        self.tab_predict = self.tabview.add("2. Predicción de Probabilidad")

        self.setup_train()
        self.setup_predict()

    def setup_train(self):
        ctk.CTkLabel(self.tab_train, text="¿Qué desea predecir?", font=("Arial", 16)).pack(pady=10)
        self.target_menu = ctk.CTkOptionMenu(self.tab_train, values=["Accidentes", "Calidad Aire", "Emergencias", "Tráfico"])
        self.target_menu.pack(pady=10)

        ctk.CTkLabel(self.tab_train, text="Algoritmo:").pack(pady=5)
        self.algo_menu = ctk.CTkOptionMenu(self.tab_train, values=["Random Forest", "Decision Tree", "SVM"])
        self.algo_menu.pack(pady=10)

        self.btn_train = ctk.CTkButton(self.tab_train, text="Entrenar Modelo", fg_color="#2ecc71", command=self.run_train)
        self.btn_train.pack(pady=20)

        self.log_train = ctk.CTkTextbox(self.tab_train, height=200, fg_color="#1a1a1a", text_color="#00ff00")
        self.log_train.pack(padx=20, pady=10, fill="both")

    def setup_predict(self):
        frame_input = ctk.CTkFrame(self.tab_predict)
        frame_input.pack(padx=20, pady=20, fill="x")

        ctk.CTkLabel(frame_input, text="Fecha (DD/MM/AAAA):").grid(row=0, column=0, padx=10, pady=10)
        self.ent_fecha = ctk.CTkEntry(frame_input, placeholder_text="Ej: 15/01/2024")
        self.ent_fecha.grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(frame_input, text="Distrito:").grid(row=1, column=0, padx=10, pady=10)
        self.combo_dist = ctk.CTkOptionMenu(frame_input, values=[f"{k}-{v}" for k,v in self.distritos.items()])
        self.combo_dist.grid(row=1, column=1, padx=10, pady=10)

        self.btn_refresh = ctk.CTkButton(self.tab_predict, text="Actualizar Modelos", command=self.update_models)
        self.btn_refresh.pack(pady=5)
        
        self.model_selector = ctk.CTkOptionMenu(self.tab_predict, values=["Entrene un modelo primero"])
        self.model_selector.pack(pady=10)

        self.btn_pred = ctk.CTkButton(self.tab_predict, text="Calcular Probabilidad", fg_color="#e67e22", command=self.predict)
        self.btn_pred.pack(pady=20)

        self.lbl_res = ctk.CTkLabel(self.tab_predict, text="Probabilidad: --%", font=("Arial", 30, "bold"))
        self.lbl_res.pack(pady=20)

        self.log_pred = ctk.CTkTextbox(self.tab_predict, height=150)
        self.log_pred.pack(padx=20, pady=10, fill="both")
        self.update_models()

    def update_models(self):
        if os.path.exists("modelos_guardados"):
            modelos = [f for f in os.listdir("modelos_guardados") if f.endswith(".pkl")]
            if modelos:
                self.model_selector.configure(values=modelos)
                self.model_selector.set(modelos[0])

    def run_train(self):
        target = self.target_menu.get()
        algo = self.algo_menu.get()
        self.btn_train.configure(state="disabled")

        def task():
            try:
                self.log_train.insert("end", f"> Entrenando {algo} para {target}...\n")
                if algo == "Random Forest": acc = entrenamiento_random_forest(target)
                elif algo == "Decision Tree": acc = entrenamiento_arbol_de_decision(target)
                else: acc = entrenamiento_svm(target)
                
                self.log_train.insert("end", f"> Éxito. Precisión: {acc:.2%}\n")
                self.update_models()
            except Exception as e:
                self.log_train.insert("end", f"> Error: {e}\n")
            finally:
                self.btn_train.configure(state="normal")

        threading.Thread(target=task).start()

    def predict(self):
        try:
            modelo_nombre = self.model_selector.get()
            fecha = datetime.strptime(self.ent_fecha.get(), "%d/%m/%Y")
            distrito = int(self.combo_dist.get().split("-")[0])
            
            modelo = joblib.load(f"modelos_guardados/{modelo_nombre}")
            
            # Preparar datos: dia, mes, año, codigo_de_distrito
            X_input = pd.DataFrame([[fecha.day, fecha.month, fecha.year, distrito]], 
                                   columns=['dia', 'mes', 'año', 'codigo_de_distrito'])
            
            # Obtener probabilidad de la clase 1 (Riesgo alto)
            prob = modelo.predict_proba(X_input)[0][1]
            
            self.lbl_res.configure(text=f"Probabilidad: {prob:.2%}")
            color = "red" if prob > 0.6 else "yellow" if prob > 0.3 else "green"
            self.lbl_res.configure(text_color=color)
            
            self.log_pred.insert("end", f"Predicción con {modelo_nombre} para {self.ent_fecha.get()}: {prob:.2%}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Verifique la fecha y el modelo: {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()