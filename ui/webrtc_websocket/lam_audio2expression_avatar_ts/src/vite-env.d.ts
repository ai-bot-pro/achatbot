/// <reference types="vite/client" />

interface ImportMeta {
    readonly env: ImportMetaEnv;
}

interface ImportMetaEnv {
    readonly VITE_SERVER_URL: string;
    // 可以在这里添加更多的环境变量
    // readonly VITE_APP_TITLE: string;
    // readonly VITE_API_URL: string;
    // ...
}