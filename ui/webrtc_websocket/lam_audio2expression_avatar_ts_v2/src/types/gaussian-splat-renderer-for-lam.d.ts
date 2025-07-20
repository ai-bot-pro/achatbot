declare module "gaussian-splat-renderer-for-lam" {
    export class GaussianSplatRenderer {
        static getInstance(
            container: HTMLDivElement,
            assetsPath: string,
            options: {
                getChatState: () => string;
                getExpressionData: () => any;
                backgroundColor: string;
            }
        ): Promise<GaussianSplatRenderer>;
    }
}