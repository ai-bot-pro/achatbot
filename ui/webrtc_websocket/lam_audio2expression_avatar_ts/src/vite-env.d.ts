/// <reference types="vite/client" />

interface ImportMeta {
    readonly env: ImportMetaEnv;
}

interface ImportMetaEnv {
    readonly VITE_SERVER_URL: string;
    readonly VITE_AVATAR_PATH: string;
    readonly VITE_PROTOBUF_PATH: string;
    // 可以在这里添加更多的环境变量
    // ...
}