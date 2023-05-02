<template>
    <header>
        <div class="flex flex-row text-center items-center h-16 content-center shadow-lg">
            <div class="basis-1/12">
                <router-link to="/"><span class="h-full">首页</span></router-link>
            </div>
            <div class="flex-auto text-xl">
                基于多尺度特征融合的高速公路违章行驶车辆识别与处理系统
            </div>
            <div class="flex-none w-32">
                <button class="rounded-full shadow-lg w-10 h-10 ring ring-blue-300 hover:ring-blue-500">
                    <el-icon class="w-10 h-10">
                        <User/>
                    </el-icon>
                </button>
            </div>
        </div>
    </header>
</template>

<script setup>
import {User} from "@element-plus/icons-vue";
</script>
