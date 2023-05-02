import { createApp } from 'vue'
import App from './App.vue'
import router from './utils/router'
import './style.css'


const app = createApp(App)

app.use(router)

app.mount('#app')
