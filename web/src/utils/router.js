import {createRouter, createWebHistory} from "vue-router";
import Index from "@/views/Index.vue";

const routes = [
	{
		path: '/',
		name: 'index',
		component: Index
	}
]

const router = createRouter({
	history: createWebHistory(),
	routes
})

// utils.beforeEach((to,from,next)=>{
// //需要授权且用户没有登录
// 	if(to.meta.requestAuth&&!store.state.user.is_login){
// 		next({name:"user_account_login"});
// 	}else{
// 		next();
// 	}
// })

export default router