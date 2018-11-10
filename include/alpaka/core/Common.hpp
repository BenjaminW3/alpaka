/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Debug.hpp>

#include <boost/predef/version_number.h>

// Boost.Uuid errors with VS2017 when intrin.h is not included
#if defined(_MSC_VER) && _MSC_VER >= 1910
    #include <intrin.h>
#endif

//-----------------------------------------------------------------------------
// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
    #define BOOST_COMP_CLANG_CUDA BOOST_COMP_CLANG
#else
    #define BOOST_COMP_CLANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// Boost does not yet correctly identify clang when compiling CUDA code.
// After explicitly including <boost/config.hpp> we can safely undefine some of the wrong settings.
#if BOOST_COMP_CLANG_CUDA
    #include <boost/config.hpp>
    #undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

//-----------------------------------------------------------------------------
// Boost disables variadic templates for nvcc (in some cases because it was buggy).
// However, we rely on it being enabled.
// After explicitly including <boost/config.hpp> we can safely undefine the wrong setting.
#if BOOST_COMP_NVCC
    #include <boost/config.hpp>
    #undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif


#ifdef ALPAKA_ACC_CPU_B_SEQ_T_COROUTINES_ENABLED
#include <experimental/coroutine>

namespace alpaka
{
    struct coreturn {
        struct promise_type {
            friend struct coreturn;
            promise_type() {
                //std::cout << "Promise created" << std::endl;
            }
            ~promise_type() {
                //std::cout << "Promise died" << std::endl;
            }
            [[ noreturn ]] void unhandled_exception() {
                std::exit(1);
            }
            auto get_return_object() {
                //std::cout << "Send back a coreturn" << std::endl;
                return coreturn{handle_type::from_promise(*this)};
            }
            auto initial_suspend() {
                //std::cout << "Started the coroutine, don't stop now!" << std::endl;
                return std::experimental::suspend_never{};
                //lazy: return std::experimental::suspend_always{};
            }
            auto final_suspend() {
                //std::cout << "Finished the coro" << std::endl;
                return std::experimental::suspend_always{};
            }
            void return_void() {}
        };

        friend struct promise_type;
        using handle_type = std::experimental::coroutine_handle<promise_type>;

        coreturn(handle_type h)
        : coro(h) {
            //std::cout << "Created a coreturn wrapper object" << std::endl;
        }
        coreturn(const coreturn &) = delete;
        coreturn(coreturn &&s)
        : coro(s.coro) {
            //std::cout << "Coreturn wrapper moved" << std::endl;
            s.coro = nullptr;
        }
        ~coreturn() {
            //std::cout << "Coreturn wrapper gone" << std::endl;
            if(coro){
                coro.destroy();
            }
        }
        coreturn &operator = (const coreturn &) = delete;
        coreturn &operator = (coreturn &&s) {
            coro = s.coro;
            s.coro = nullptr;
            return *this;
        }
        void wait() {
            //std::cout << "We got asked to finish..." << std::endl;
            if(!this->coro.done()){
                this->coro.resume();
            }
            return;
        }
    protected:
        handle_type coro;
    };
}
#define ALPAKA_FN_RET alpaka::coreturn
#else
#define ALPAKA_FN_RET void
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FN_ACC or ALPAKA_FN_HOST_ACC.
//!
//! Usage:
//! ALPAKA_FN_ACC
//! auto add(std::int32_t a, std::int32_t b)
//! -> std::int32_t;
#if BOOST_LANG_CUDA || BOOST_LANG_HIP
    #if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) || defined(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
        #define ALPAKA_FN_ACC __device__
    #else
        #define ALPAKA_FN_ACC __device__ __host__
    #endif
    #define ALPAKA_FN_HOST_ACC __device__ __host__
    #define ALPAKA_FN_HOST __host__
#else
    #define ALPAKA_FN_ACC
    #define ALPAKA_FN_HOST_ACC
    #define ALPAKA_FN_HOST
#endif

//-----------------------------------------------------------------------------
//! Disable nvcc warning:
//! 'calling a __host__ function from __host__ __device__ function.'
//!
//! Usage:
//! ALPAKA_NO_HOST_ACC_WARNING
//! ALPAKA_FN_HOST_ACC function_declaration()
//!
//! WARNING: Only use this method if there is no other way.
//! Most cases can be solved by #if BOOST_ARCH_PTX or #if BOOST_LANG_CUDA.
#if (BOOST_LANG_CUDA && !BOOST_COMP_CLANG_CUDA) \
  || BOOST_LANG_HIP
    #if BOOST_COMP_MSVC
        #define ALPAKA_NO_HOST_ACC_WARNING\
            __pragma(hd_warning_disable)
    #else
        #define ALPAKA_NO_HOST_ACC_WARNING\
            _Pragma("hd_warning_disable")
    #endif
#else
    #define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! Macro defining the inline function attribute.
#if BOOST_LANG_CUDA || BOOST_LANG_HIP
    #define ALPAKA_FN_INLINE __forceinline__
#else
    #define ALPAKA_FN_INLINE inline
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in global accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_ACC_MEM_GLOBAL int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_ACC_MEM_GLOBAL int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if (BOOST_LANG_CUDA && BOOST_ARCH_PTX) || (BOOST_LANG_HIP && (BOOST_ARCH_HSA || BOOST_ARCH_PTX))
    #define ALPAKA_STATIC_ACC_MEM_GLOBAL __device__
#else
    #define ALPAKA_STATIC_ACC_MEM_GLOBAL
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in constant accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_ACC_MEM_CONSTANT int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_ACC_MEM_CONSTANT int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if (BOOST_LANG_CUDA && BOOST_ARCH_PTX) || (BOOST_LANG_HIP && (BOOST_ARCH_HSA || BOOST_ARCH_PTX))
    #define ALPAKA_STATIC_ACC_MEM_CONSTANT __constant__
#else
    #define ALPAKA_STATIC_ACC_MEM_CONSTANT
#endif
