use std::num::NonZeroU64;

use cmli::xva::{XvaCategory, XvaType};
use lccc_targets::properties::target::Target;
use lxca::ir::{
    constant::ConstantPool,
    types::{IntType, Type},
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SizeAlign {
    pub size: u64,
    pub align: NonZeroU64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypeLayout {
    pub size_align: SizeAlign,
    pub category: XvaCategory,
    pub fields: Fields,
}

impl TypeLayout {
    pub const fn scalar_layout(width: u16, category: XvaCategory, target: &Target) -> Self {
        let max_align = target.primitive_layout.as_ref().max_int_align as u64;
        assert!(max_align.is_power_of_two());
        let raw_size = ((width + 7) / 8) as u64;

        let align = if raw_size < max_align {
            raw_size.next_power_of_two()
        } else {
            max_align
        };
        let size = (raw_size + (align - 1)) & !(align - 1);

        // SAFETY: We know `align` is a power of two, and `next_power_of_two` could not have overflowed as the maximum possibly size is 8192.
        let align = unsafe { NonZeroU64::new_unchecked(align) };

        Self {
            size_align: SizeAlign { size, align },
            category,
            fields: Fields::Scalar,
        }
    }

    pub const fn xva_type(&self) -> XvaType {
        XvaType {
            size: self.size_align.size,
            align: self.size_align.align.get(),
            category: self.category,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Fields {
    Scalar,
    Struct(StructFields),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct StructFields {
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Field {
    pub offset: u64,
    pub layout: TypeLayout,
    pub field_kind: FieldKind,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum FieldKind {
    /// A Layout Field
    Layout,
    /// A Bit field
    Bitfield {
        /// Specifies the offset without the byte that the bitfield begins
        bitoffset: u8,
        /// Specifies the width of the bitfield
        field_witdh: u16,
    },
}

pub fn layout_type<'ir>(
    ty: &Type<'ir>,
    constants: &ConstantPool<'ir>,
    targ: &Target,
) -> TypeLayout {
    match ty.body(constants) {
        lxca::ir::types::TypeBody::Interned(_) => unreachable!(),
        lxca::ir::types::TypeBody::Named(name) => todo!("named type {:?}", name.get(constants)),
        lxca::ir::types::TypeBody::Integer(int_type) => {
            TypeLayout::scalar_layout(int_type.width, XvaCategory::Int, targ)
        }
        lxca::ir::types::TypeBody::Char(width) => {
            TypeLayout::scalar_layout(*width, XvaCategory::Int, targ)
        }
        lxca::ir::types::TypeBody::Pointer(_) => TypeLayout::scalar_layout(
            targ.primitive_layout.int_layout.short_pointer_width,
            XvaCategory::Int,
            targ,
        ),
        lxca::ir::types::TypeBody::Function(_) => panic!("Cannot layout a function type"),
        lxca::ir::types::TypeBody::Void => panic!("Cannot layout void"),
    }
}
