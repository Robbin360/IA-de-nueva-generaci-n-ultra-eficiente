import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const PORT = 3000;

async function startServer() {
  const app = express();
  app.use(express.json());

  // Health Check
  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', engine: 'Aethel-Genesis UI' });
  });

  // Chat Endpoint - Proxies to the conceptual Aethel backend
  app.post('/api/chat', (req, res) => {
    try {
      const { message } = req.body;
      
      // Simulate processing time from the Rust/Triton engine
      setTimeout(() => {
        // En una implementación real, aquí se llamaría al binario de Rust o Python.
        // Para propósitos de esta UI de demostración, respondemos simulando el motor.
        const responseText = `[Aethel V3 - Kernel Triton Activo]\nHe procesado tu mensaje: "${message}". La arquitectura Matemática (MoE Load Balancing, SwiGLU, GQA) está operando correctamente en simulación. Para uso en producción, compila el demonio en Rust (\`exoesqueleto_rust\`).`;
        
        res.json({ reply: responseText });
      }, 800);

    } catch (error: any) {
      console.error('Error en /api/chat:', error);
      res.status(500).json({ error: 'Error interno de comunicación con el exoesqueleto.' });
    }
  });

  // Vite development middleware vs Static Production
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
