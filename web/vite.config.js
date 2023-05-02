import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import {ElementPlusResolver} from 'unplugin-vue-components/resolvers'
import IconsResolver from 'unplugin-icons/resolver'
import path from 'path'
import Icons from 'unplugin-icons/vite'
import Inspect from 'vite-plugin-inspect'
import {fileURLToPath} from "node:url";
// https://vitejs.dev/config/

const pathSrc = path.resolve(__dirname, 'src')

export default defineConfig({
	resolve: {
		alias: {
			'@': fileURLToPath(new URL('./src', import.meta.url)),
		},
	},
	plugins: [
		vue(),
		AutoImport({
			imports: ['vue'],
			resolvers: [
				ElementPlusResolver(),
				IconsResolver({
					prefix: 'Icon',
				}),
			],

			dts: path.resolve(pathSrc, 'auto-imports.d.ts'),
		}),
		Components({
			resolvers: [
				IconsResolver({
					enabledCollections: ['ep'],
				}),
				ElementPlusResolver(),
			],

			dts: path.resolve(pathSrc, 'components.d.ts'),
		}),

		Icons({
			autoInstall: true,
		}),

		Inspect(),
	],
	build: {
		rollupOptions: {
			output: {
				manualChunks(id) {
					if (id.includes('node_modules')) {
						return id
							.toString()
							.split('node_modules/')[1]
							.split('/')[0]
							.toString()
					}
				},
			},
		},
	}
})
