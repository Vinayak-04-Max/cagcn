const http = require("http");
const { URL } = require("url");
const path = require("path");
const fs = require("fs");

const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT || 3000);
const ML_API_BASE_URL = process.env.ML_API_BASE_URL || "http://127.0.0.1:8000";
const PUBLIC_DIR = path.join(process.cwd(), "public");

const routes = {
  "GET /api/health": handleHealth,
  "GET /api/wafers": handleRandomWafers,
  "POST /api/predict": handlePredict
};

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml"
};

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const routeKey = `${req.method} ${url.pathname}`;

    if (routes[routeKey]) {
      await routes[routeKey](req, res, url);
      return;
    }

    if (req.method === "GET") {
      await serveStaticAsset(res, url.pathname);
      return;
    }

    sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    console.error("Unexpected server error:", error);
    sendJson(res, 500, {
      error: "Unexpected server error",
      details: error.message
    });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Wafer dashboard running at http://localhost:${PORT}`);
  console.log(`Proxying ML requests to ${ML_API_BASE_URL}`);
});

function handleHealth(_req, res) {
  sendJson(res, 200, {
    status: "ok",
    frontend: "ready",
    mlApiBaseUrl: ML_API_BASE_URL
  });
}

async function handleRandomWafers(_req, res, url) {
  const count = Number(url.searchParams.get("count") || 10);
  const payload = await proxyJson(`${ML_API_BASE_URL}/wafers/random?count=${count}`);
  sendJson(res, 200, payload);
}

async function handlePredict(req, res) {
  const body = await readJsonBody(req);
  const payload = await proxyJson(`${ML_API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  sendJson(res, 200, payload);
}

async function proxyJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};

  if (!response.ok) {
    const error = new Error(payload.error || "Upstream request failed");
    error.statusCode = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";

    req.on("data", (chunk) => {
      raw += chunk;
    });

    req.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(new Error("Invalid JSON body"));
      }
    });

    req.on("error", reject);
  });
}

async function serveStaticAsset(res, pathname) {
  const normalizedPath = pathname === "/" ? "/index.html" : pathname;
  const safePath = path.normalize(normalizedPath).replace(/^(\.\.[/\\])+/, "");
  const assetPath = path.join(PUBLIC_DIR, safePath);

  if (!assetPath.startsWith(PUBLIC_DIR)) {
    sendJson(res, 403, { error: "Forbidden" });
    return;
  }

  try {
    const stats = await fs.promises.stat(assetPath);
    const filePath = stats.isDirectory() ? path.join(assetPath, "index.html") : assetPath;
    const extension = path.extname(filePath).toLowerCase();
    const content = await fs.promises.readFile(filePath);

    res.writeHead(200, {
      "Content-Type": contentTypes[extension] || "application/octet-stream",
      "Cache-Control": "no-store"
    });
    res.end(content);
  } catch (_error) {
    sendJson(res, 404, { error: "Asset not found" });
  }
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8"
  });
  res.end(JSON.stringify(payload));
}
