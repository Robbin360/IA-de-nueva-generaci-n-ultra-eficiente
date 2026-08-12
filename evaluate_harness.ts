import fs from 'fs';

console.log("==================================================");
console.log("   AETHEL NOVA: EVALUATION HARNESS (NODEJS)       ");
console.log("==================================================");
console.log("[*] Iniciando entorno de evaluación tipo LM-Eval local...");

async function downloadDataset(url: string, filename: string) {
    console.log(`[*] Descargando dataset desde ${url}...`);
    const res = await fetch(url);
    const text = await res.text();
    fs.writeFileSync(filename, text);
    console.log(`[+] Dataset guardado en ${filename}`);
}

async function run() {
    // GSM8K sample (solo unas pocas preguntas para demostración)
    const gsm8kUrl = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl";
    
    try {
        if (!fs.existsSync('gsm8k_test.jsonl')) {
            await downloadDataset(gsm8kUrl, 'gsm8k_test.jsonl');
        }
        
        const lines = fs.readFileSync('gsm8k_test.jsonl', 'utf-8').split('\n').filter(Boolean).slice(0, 5); // Tomamos 5 para el test rápido
        
        console.log(`\n[*] Evaluando GSM8K (Razonamiento Matemático) - ${lines.length} samples...`);
        let correct = 0;
        
        for (let i = 0; i < lines.length; i++) {
            const item = JSON.parse(lines[i]);
            console.log(`\n--- Pregunta ${i+1} ---`);
            console.log(`Q: ${item.question}`);
            
            // Aquí llamaríamos a nuestro modelo. Como el modelo Nova actual 
            // está inicializado con pesos aleatorios y solo entrenado en Shakespeare,
            // simulamos el paso de inferencia y mostramos que el pipeline funciona.
            
            const start = Date.now();
            let dummyAnswer = "";
            let time = 0;
            
            try {
                const response = await fetch("http://localhost:3000/v1/chat/completions", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        model: "aethel-quantum",
                        messages: [{ role: "user", content: item.question }],
                        temperature: 0.1
                    })
                });
                const data = await response.json();
                dummyAnswer = data.choices[0].message.content;
                time = Date.now() - start;
            } catch (err: any) {
                dummyAnswer = "Error connecting to model: " + err.message;
            }
            
            console.log(`Aethel-Quantum (Pre-trained): \n${dummyAnswer}\n(${time}ms)`);
            console.log(`Esperado: ${item.answer.split("#### ")[1]}`);
            
            // Check si la respuesta generada contiene el número esperado (muy ingenuo, pero sirve para el demo)
            if (dummyAnswer.includes(item.answer.split("#### ")[1])) {
                correct++;
                console.log("Resultado: ✅ PASS");
            } else {
                console.log("Resultado: ❌ FAIL");
            }
        }
        
        console.log(`\n=== RESULTADOS MMLU/GSM8K ===`);
        console.log(`GSM8K Accuracy: ${(correct / lines.length) * 100}%`);
        console.log(`Nota: El modelo actual de JS puro requiere entrenamiento de pre-entrenamiento (Pre-training) de meses en un clúster de GPUs reales para responder correctamente. La arquitectura soporta la propagación, pero los pesos actuales son aleatorios.`);
        
    } catch(e) {
        console.error("Error en evaluación:", e);
    }
}

run();
