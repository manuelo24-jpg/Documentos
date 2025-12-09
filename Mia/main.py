from ultralytics import YOLO
import cv2
import os


def main():
    # ========================================================================
    # 1. CARGAR EL MODELO BASE PRE-ENTRENADO
    # ========================================================================
    # 'n' es nano (rápido), 'seg' habilita la segmentación (máscaras)
    print("Cargando modelo...")
    model = YOLO("yolo11n-seg.pt")
    
    # ========================================================================
    # 2. ENTRENAR EL MODELO (AFINAR / FINE-TUNING)
    # ========================================================================
    # Asegúrate de que la ruta al data.yaml sea absoluta o correcta relativa
    print("Iniciando entrenamiento...")
    results = model.train(
        data="datasets/masas_agua/data.yaml",  # Ruta a tu dataset
        epochs=10,          # Reducido a 10 para la clase (para que dé tiempo)
        imgsz=640,          # Tamaño de imagen
        plots=True,         # Generar gráficas de pérdida
        device='cpu',       # Forzar CPU si no tienen CUDA configurado (opcional)
        project='clase_yolo',  # Carpeta donde se guardan resultados
        name='agua_run'     # Nombre del experimento
    )
    
    # ========================================================================
    # 3. CARGAR EL MODELO QUE ACABAMOS DE ENTRENAR (BEST WEIGHTS)
    # ========================================================================
    # La ruta dependerá de dónde se creó la carpeta 'clase_yolo'
    best_weight_path = os.path.join('clase_yolo', 'agua_run', 'weights', 'best.pt')
    print(f"Cargando el modelo entrenado desde: {best_weight_path}")
    tuned_model = YOLO(best_weight_path)
    
    # ========================================================================
    # 4. INFERENCIA (PREDICCIÓN) EN NUEVAS IMÁGENES
    # ========================================================================
    # Usamos una imagen de prueba del dataset
    test_img_path = "datasets/masas_agua/images/test/imagen_1.jpg"  # CAMBIAR POR UNA IMAGEN REAL
    
    if os.path.exists(test_img_path):
        print(f"Realizando predicción en: {test_img_path}")
        results = tuned_model(test_img_path)
        
        # ====================================================================
        # 5. MOSTRAR RESULTADOS
        # ====================================================================
        for result in results:
            # Guardar la imagen en disco
            result.save(filename="resultado_prediccion.jpg")
            print("Resultado guardado en: resultado_prediccion.jpg")
            
            # Mostrar en una ventana emergente (típico de OpenCV)
            im_array = result.plot()  # plot() dibuja las máscaras y cajas en la imagen
            cv2.imshow("Deteccion de Agua", im_array)
            cv2.waitKey(0)  # Esperar a que se pulse una tecla
            cv2.destroyAllWindows()
    else:
        print(f"❌ ERROR: No se encontró la imagen de prueba: {test_img_path}")
        print("Por favor, verifica la ruta de la imagen.")


if __name__ == '__main__':
    # En Windows es necesario proteger el entry point para multiproceso
    main()